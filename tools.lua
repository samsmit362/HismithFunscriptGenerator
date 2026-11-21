
-- tools.lua - auxiliary tools for the Hismith Funscript Generator extension
-- (scale / move / align / stretch / insert).
--
-- Loaded by main.lua via require("tools") (OFS adds <extdir>/?.lua to
-- package.path, so this file resolves next to main.lua).
--
-- Entry points (all optional from main.lua's side - it guards each call):
--   tools_init()      - runs once when the extension is loaded
--   tools_update(delta) - runs every frame, even when the window is closed
--   tools_gui()       - runs only when the window is open (drawn inside
--                       a CollapsingHeader in main.lua)

-- globals
Settings = {}
Settings.Min = 0
Settings.Max = 100
Settings.Move = 10
Settings.IgnoreMax = false
Settings.IgnoreMin = false
Settings.AlignEdges = true
Settings.AlignRawPath = true

function tools_init()
    -- this runs once when loading the extension
    print("tools.lua: initialized")
end

function tools_update(delta)
    -- this runs even when the gui is closed
end

function tools_gui()
    -- this only runs when the window is open
    ofs.Text("Scale Settings")

	Settings.Min = ofs.Input("Min", Settings.Min, 1)
	Settings.Max = ofs.Input("Max", Settings.Max, 1)
	Settings.Move = ofs.Input("MoveVal", Settings.Move, 1)
	Settings.IgnoreMin = ofs.Checkbox("Ignore Min", Settings.IgnoreMin)
	Settings.IgnoreMax = ofs.Checkbox("Ignore Max", Settings.IgnoreMax)
	Settings.AlignEdges = ofs.Checkbox("Align to range edges (not 0-100)", Settings.AlignEdges)
	Settings.AlignRawPath = ofs.Checkbox("Align: weight by raw path length (ofs_result_raw.csv step column)", Settings.AlignRawPath)

    if ofs.Button("Apply") then
		Settings.Min = clamp(Settings.Min, 0, 100)
		Settings.Max = clamp(Settings.Max, 0, 100)
		print("Settings.Min: " .. tostring(Settings.Min) .. " Settings.Max: " .. tostring(Settings.Max))
		if Settings.Max > Settings.Min then
			scale()
		end
    end
    ofs.Tooltip(
        "Rescale the selected points vertically: each monotone (single-direction) run of selected points is linearly remapped\n" ..
		"so its endpoints land on the Min / Max range above (or on the neighboring unselected points' positions when 'Ignore Min' / 'Ignore Max' is off).\n" ..
		"The run's direction is preserved; runs shorter than 10 units are left untouched. Min / Max are clamped to 0-100 first.")

	if ofs.Button("Move Vertical By Position") then
		move(Settings.Move)
    end
	ofs.Tooltip(
        "Shift every selected point vertically by the 'MoveVal' amount (pos = pos + MoveVal).\n" ..
		"Points that would cross pos 100 are clamped to 100 and extra points are inserted\n" ..
		"at the crossing so the curve stays continuous.")

	if ofs.Button("Align") then
		align()
    end
	ofs.Tooltip(
        "Redistribute the selected points onto a single monotone stroke.\n" ..
		"Each segment's weight (the raw per-frame path length from ofs_result_raw.csv\n" ..
		"when 'Align: weight by raw path length' is on, otherwise the 0-100 position delta)\n" ..
		"sets how far the stroke advances.\n" ..
		"Endpoints: 'Align to range edges' uses the first / last selected point's positions;\n" ..
		"otherwise the stroke runs 0->100 or 100->0 based on the first segment's direction.")

	if ofs.Button("Stretch Selection To Current Video Time") then
		stretch_to_current_time()
	   end
	ofs.Tooltip(
        "Stretch the selected points along the time axis so the range spans from the current video time (playhead)\n" ..
		"to the opposite edge of the selection: the edge closest to the playhead stays pinned, the other edge moves to the playhead.\n" ..
		"Every point keeps its relative position in the span, and its pos is taken from\n" ..
		"the original curve at the matching original time (snapped to the nearest video frame).")

	if ofs.Button("Insert With Stretch") then
		insert_with_stretch()
	   end
	ofs.Tooltip(
	       "Insert the copied points (clipboard) at the current video time, stretching each\n" ..
		"monotonic segment (rising/falling) of the copied curve onto the corresponding segment\n" ..
		"of the current timeline. Each copied extremum is mapped to the matching extremum type\n" ..
		"(peak->peak, valley->valley) on the timeline after the playhead. If the timeline has\n" ..
		"fewer extrema, only that many segments of the copied curve are stretched and inserted.")
end

function move(shift_val)
	local script = ofs.Script(ofs.ActiveIdx())
	local actionsCount = #script.actions
	print("actionsCount: " .. tostring(actionsCount))
	local new_idx = 0
	updated_pos = {}

    for idx=1,actionsCount,1 do
		new_idx = new_idx + 1
		updated_pos[new_idx] = {}
        if script.actions[idx].selected then
			if script.actions[idx].pos + shift_val > 100 then
				if idx > 1 and script.actions[idx-1].pos + shift_val < 100 then
					updated_pos[new_idx].at = script.actions[idx-1].at + (((100 - script.actions[idx-1].pos - shift_val)*(script.actions[idx].at - script.actions[idx-1].at))/(script.actions[idx].pos - script.actions[idx-1].pos))
					updated_pos[new_idx].pos = 100
					updated_pos[new_idx].selected = true
					print("1 add left point idx: " .. tostring(idx) .. " new_idx: " .. tostring(new_idx) .. " old_at: " .. tostring(script.actions[idx].at) .. " prev_idx_at: " .. tostring(script.actions[idx-1].at) .. " new_at: " .. tostring(updated_pos[new_idx].at))

					if idx < actionsCount and script.actions[idx+1].pos + shift_val < 100 then
						new_idx = new_idx + 1
						updated_pos[new_idx] = {}
						updated_pos[new_idx].at = script.actions[idx+1].at - (((100 - script.actions[idx+1].pos - shift_val)*(script.actions[idx+1].at - script.actions[idx].at))/(script.actions[idx].pos - script.actions[idx+1].pos))
						updated_pos[new_idx].pos = 100
						updated_pos[new_idx].selected = true
						print("1 add right point idx: " .. tostring(idx) .. " new_idx: " .. tostring(new_idx) .. " old_at: " .. tostring(script.actions[idx].at) .. " next_idx_at: " .. tostring(script.actions[idx+1].at) .. " new_at: " .. tostring(updated_pos[new_idx].at))
						script.actions:add(Action.new(updated_pos[new_idx].at, updated_pos[new_idx].pos, updated_pos[new_idx].selected))
					end
				else
					if idx < actionsCount and script.actions[idx+1].pos + shift_val < 100 then
						updated_pos[new_idx].at = script.actions[idx+1].at - (((100 - script.actions[idx+1].pos - shift_val)*(script.actions[idx+1].at - script.actions[idx].at))/(script.actions[idx].pos - script.actions[idx+1].pos))
						updated_pos[new_idx].pos = 100
						updated_pos[new_idx].selected = true
						print("2 add right point idx: " .. tostring(idx) .. " new_idx: " .. tostring(new_idx) .. " old_at: " .. tostring(script.actions[idx].at) .. " next_idx_at: " .. tostring(script.actions[idx+1].at) .. " new_at: " .. tostring(updated_pos[new_idx].at))
					else
						updated_pos[new_idx].at = script.actions[idx].at
						updated_pos[new_idx].pos = 100
						updated_pos[new_idx].selected = true
					end
				end
			else
				updated_pos[new_idx].at = script.actions[idx].at
				updated_pos[new_idx].pos = script.actions[idx].pos + shift_val
				updated_pos[new_idx].selected = true
			end
		else
			updated_pos[new_idx].at = script.actions[idx].at
			updated_pos[new_idx].pos = script.actions[idx].pos
			updated_pos[new_idx].selected = script.actions[idx].selected
		end
    end

	actionsCount = #updated_pos
	for idx=1,actionsCount,1 do
		print("move idx: " .. tostring(idx) .. " at: " .. tostring(updated_pos[idx].at) .. " pos: " .. tostring(updated_pos[idx].pos))
		script.actions[idx].at = updated_pos[idx].at
		script.actions[idx].pos = updated_pos[idx].pos
		script.actions[idx].selected = updated_pos[idx].selected
	end

    script:commit()
end


function update_move(updated_pos, script, start_idx, end_idx)
	local actionsCount = #script.actions
	local start_pos = script.actions[start_idx].pos
	local end_pos = script.actions[end_idx].pos
	local min_pos = -1
	local max_pos = -1

	print("update_move: start_idx_and_pos(" .. tostring(start_idx) .. "," .. tostring(start_pos) .. ")" .. " end_idx_and_pos(" .. tostring(end_idx) .. "," .. tostring(end_pos) .. ") ...")

	if start_pos < end_pos then
		if start_idx == 1 then
			min_pos = Settings.Min
		elseif script.actions[start_idx - 1].selected then
			min_pos = Settings.Min
		elseif not Settings.IgnoreMin then
			min_pos = script.actions[start_idx].pos
		else
			min_pos = Settings.Min
		end

		if end_idx == actionsCount then
			max_pos = Settings.Max
		elseif script.actions[end_idx + 1].selected then
			max_pos = Settings.Max
		elseif not Settings.IgnoreMax then
			max_pos = script.actions[end_idx].pos
		else
			max_pos = Settings.Max
		end

		print("min_pos: " .. tostring(min_pos) .. " max_pos: " .. tostring(max_pos))

		if max_pos >= min_pos + 10 then
			for idx=start_idx,end_idx do
				updated_pos[idx] = min_pos + ((max_pos - min_pos) * (script.actions[idx].pos - start_pos)) // (end_pos - start_pos)
			end
		end
	else
		if start_idx == 1 then
			max_pos = Settings.Max
		elseif script.actions[start_idx - 1].selected then
			max_pos = Settings.Max
		elseif not Settings.IgnoreMax then
			max_pos = script.actions[start_idx].pos
		else
			max_pos = Settings.Max
		end

		if end_idx == actionsCount then
			min_pos = Settings.Min
		elseif script.actions[end_idx + 1].selected then
			min_pos = Settings.Min
		elseif not Settings.IgnoreMin then
			min_pos = script.actions[end_idx].pos
		else
			min_pos = Settings.Min
		end

		print("min_pos: " .. tostring(min_pos) .. " max_pos: " .. tostring(max_pos))

		if max_pos >= min_pos + 10 then
			for idx=start_idx,end_idx do
				updated_pos[idx] = max_pos - ((max_pos - min_pos) * (start_pos - script.actions[idx].pos)) // (start_pos - end_pos)
			end
		end
	end
end

function scale()
	local start_idx = -1
	local end_idx = -1
	updated_pos = {}

	-- prev_direction -1 - undefined, 0 - pos_up, 1 - pos_down
	local prev_direction = -1
	local cur_direction = -1

	local prev_action = nil

    local script = ofs.Script(ofs.ActiveIdx())
	local actionsCount = #script.actions
	print("actionsCount: " .. tostring(actionsCount))

    for idx, action in ipairs(script.actions) do
        if action.selected then
			if start_idx == -1 then
				start_idx = idx
				end_idx = start_idx
				prev_action = action
			else
				if action.pos > prev_action.pos then
					cur_direction = 0
				elseif action.pos < prev_action.pos then
					cur_direction = 1
				end

				if cur_direction ~= -1 and prev_direction ~= -1 and cur_direction ~= prev_direction then
					end_idx = idx - 1
					update_move(updated_pos, script, start_idx, end_idx)
					start_idx = end_idx
				end

				if idx == actionsCount or (idx < actionsCount and not script.actions[idx + 1].selected) then
					end_idx = idx
					update_move(updated_pos, script, start_idx, end_idx)
					start_idx = end_idx
				end

				prev_direction = cur_direction
				prev_action = action
			end
		end
    end

	for idx,pos in pairs(updated_pos) do
		print("idx: " .. tostring(idx) .. " pos: " .. tostring(pos))
		script.actions[idx].pos = pos
	end

    script:commit()
end


-- Load the raw per-frame path steps from ofs_result_raw.csv (next to
-- main.lua).  The CSV is written by funscript_gen.py with columns
-- time_s,pos,step where `step` is the per-frame (ps-compensated)
-- displacement of point p in px - the same quantity the "complex" motion
-- mode integrates (pos = 100 * cumsum(step) / total).  Returns two parallel
-- arrays: raw_times[i] (time_s), raw_steps[i] (step).  Returns nil, nil if
-- the file is missing or has no usable rows.
local function load_raw_steps()
	local path = ofs.ExtensionDir() .. "\\ofs_result_raw.csv"
	local f = io.open(path, "r")
	if not f then
		print("align: raw path CSV not found: " .. path)
		return nil, nil
	end
	local times = {}
	local steps = {}
	local first = true
	for line in f:lines() do
		if first then
			first = false
		else
			local t, _p, s = line:match("([^,]+),([^,]+),([^,]+)")
			local tv, sv = (t and tonumber(t)), (s and tonumber(s))
			if tv and sv then
				times[#times + 1] = tv
				steps[#steps + 1] = sv
			end
		end
	end
	f:close()
	if #times == 0 then
		print("align: raw path CSV has no usable rows: " .. path)
		return nil, nil
	end
	return times, steps
end

-- Align: redistribute the selected range onto a monotone stroke, preserving
-- the relative "weights" of each segment.  The per-segment weight is, by
-- default (Settings.AlignRawPath), the RAW per-frame path length in px taken
-- from ofs_result_raw.csv (the `step` column - the same quantity the
-- "complex" motion mode integrates: pos = 100 * cumsum(step) / total).  This
-- makes the stroke follow the true physical trajectory rather than the
-- (already rescaled) 0-100 positions.  When the CSV is unavailable, or the
-- checkbox is off, the weight falls back to |dP| in the 0-100 space.
--
-- Two endpoint modes (toggle via Settings.AlignEdges):
--   * classic (default): stroke endpoints are A=100 / B=0 or A=0 / B=100,
--     chosen from the FIRST INTERNAL segment of the selected range
--     (orig[2] - orig[1]).  Pre-range points are NOT consulted.
--   * edges: stroke endpoints are the left and right edge of the range
--     (A = orig[1], B = orig[#orig]); direction follows automatically.
--
-- Formula applied in both modes:
--   Pi = A + (B - A) * (S_cum_i / S)
-- where S_cum_i = SUM_{j=2..i} w_j and S = S_cum_n, with w_j the per-segment
-- weight (raw px path length, or |orig[j] - orig[j-1]| as a fallback).
function align()
	local script = ofs.Script(ofs.ActiveIdx())

	-- 1) collect the selected points in script order
	local sel_idx = {}
	for idx, action in ipairs(script.actions) do
		if action.selected then
			sel_idx[#sel_idx + 1] = idx
		end
	end

	if #sel_idx == 0 then
		print("align: no selected points")
		return
	end
	if #sel_idx == 1 then
		print("align: only one selected point, nothing to align")
		return
	end

	local n = #sel_idx
	print("align: range [" .. tostring(sel_idx[1]) .. " .. " .. tostring(sel_idx[n]) .. "], points=" .. tostring(n))

	-- 2) snapshot ORIGINAL positions and times (must read before mutating)
	local orig = {}
	local orig_t = {}
	for i, idx in ipairs(sel_idx) do
		orig[i] = script.actions[idx].pos
		orig_t[i] = script.actions[idx].at
	end

	-- 3) determine A and B
	local A, B
	if Settings.AlignEdges then
		A = orig[1]
		B = orig[n]
		print(string.format("align: mode=edges  A=%.3f (left edge)  B=%.3f (right edge)", A, B))
		if A == B then
			print("align: edges equal -> no redistribution (all points stay at " .. tostring(A) .. ")")
			return
		end
	else
		-- classic: first INTERNAL segment decides direction, pre-range ignored
		local d0 = orig[2] - orig[1]
		if d0 < 0 then
			A = 100
			B = 0
			print(string.format("align: mode=classic  first segment d0=%.3f < 0 -> A=100 B=0 (monotone down)", d0))
		elseif d0 > 0 then
			A = 0
			B = 100
			print(string.format("align: mode=classic  first segment d0=%.3f > 0 -> A=0 B=100 (monotone up)", d0))
		else
			print("align: mode=classic  first segment d0=0 (flat start); defaulting to A=0 B=100 (monotone up)")
			A = 0
			B = 100
		end
	end

	-- 4) per-segment weights w_i (i = 2..n) and the total S
	--    raw mode: w_i = the physical path length (px) between the two
	--    selected points, integrated from the ofs_result_raw.csv `step`
	--    column (the "complex" motion mode trajectory);
	--    fallback: w_i = |orig[i] - orig[i-1]| (the 0-100 position delta).
	local w = {}
	local S = 0
	if Settings.AlignRawPath then
		local raw_times, raw_steps = load_raw_steps()
		if raw_times then
			local raw_n = #raw_times
			for i = 2, n do
				local t_from, t_to = orig_t[i - 1], orig_t[i]
				local lo, hi = 1, raw_n
				while hi - lo > 1 do
					local mid = (lo + hi) // 2
					if raw_times[mid] <= t_from then
						lo = mid
					else
						hi = mid
					end
				end
				local lo2, hi2 = 1, raw_n
				while hi2 - lo2 > 1 do
					local mid = (lo2 + hi2) // 2
					if raw_times[mid] <= t_to then
						lo2 = mid
					else
						hi2 = mid
					end
				end
				local sum = 0
				for k = lo2, hi2 do
					if k > lo then
						sum = sum + raw_steps[k]
					end
				end
				w[i] = sum
				S = S + sum
			end
			print(string.format("align: mode=raw  total path S=%.3f px (from ofs_result_raw.csv, %d step row(s))", S, raw_n))
		else
			for i = 2, n do
				local d = orig[i] - orig[i - 1]
				if d < 0 then d = -d end
				w[i] = d
				S = S + d
			end
			print("align: mode=raw requested but ofs_result_raw.csv unavailable -> falling back to |dP|, total S=" .. tostring(S))
		end
	else
		for i = 2, n do
			local d = orig[i] - orig[i - 1]
			if d < 0 then d = -d end
			w[i] = d
			S = S + d
		end
		print("align: mode=dp (0-100 position delta), total S=" .. tostring(S))
	end

	if S == 0 then
		print("align: S=0 (all weights identical), flattening range to midpoint " .. tostring((A + B) / 2))
		local mid = (A + B) / 2
		for i, idx in ipairs(sel_idx) do
			script.actions[idx].pos = mid
		end
		script:commit()
		return
	end

	-- 5) apply Pi = A + (B - A) * (S_cum_i / S)
	local S_cum = 0
	for i, idx in ipairs(sel_idx) do
		local target = A + (B - A) * (S_cum / S)
		target = math.floor(target * 10 + 0.5) / 10 -- round to 0.1
		if target < 0 then target = 0 end
		if target > 100 then target = 100 end

		local old = script.actions[idx].pos
		script.actions[idx].pos = target
		print(string.format("align: idx=%d old=%.3f new=%.3f cum=%.3f S=%.3f", idx, old, target, S_cum, S))

		-- advance the cumulative weight
		if i < n then
			S_cum = S_cum + (w[i + 1] or 0)
		end
	end

	script:commit()
end

-- Evaluate the original (selected) curve at a given original time using
-- linear interpolation between the two points that bracket it. `times` and
-- `positions` must be the selected points' original times and positions
-- sorted by time. Returns the clamped position in the 0-100 range.
local function curve_pos_at(times, positions, t)
	local n = #times
	if n == 0 then
		return 0
	end
	if t <= times[1] then
		return positions[1]
	end
	if t >= times[n] then
		return positions[n]
	end

	-- binary search for the bracket [times[lo], times[hi]]
	local lo, hi = 1, n
	while hi - lo > 1 do
		local mid = (lo + hi) // 2
		if times[mid] <= t then
			lo = mid
		else
			hi = mid
		end
	end

	local dt = times[hi] - times[lo]
	if dt == 0 then
		return positions[lo]
	end
	local frac = (t - times[lo]) / dt
	local pos = positions[lo] + (positions[hi] - positions[lo]) * frac
	return clamp(pos, 0, 100)
end

-- Stretch the selected fragment along the time axis so that it spans from
-- the current video time (the white vertical playhead line) to the OTHER
-- edge of the selected range, keeping every intermediate selected action at
-- its relative position inside the span.
--
-- Two cases:
--   * T >= t0: the left edge (earliest selected action) stays pinned at
--     t0, the right edge (latest selected action) is moved to T.
--   * T <  t0: the left edge is moved to T, the right edge (latest
--     selected action) stays pinned at t1.
--
-- Frame approximation: each remapped time is snapped to the nearest video
-- frame (using player.FPS()). The position of every point is then set to the
-- value of the ORIGINAL curve at the equivalent original time (the inverse of
-- the stretch formula), so the resulting curve shape stays as close as
-- possible to the original. Positions are rounded to the nearest integer
-- (0-100).
function stretch_to_current_time()
	local script = ofs.Script(ofs.ActiveIdx())

	-- 1) collect the selected points in script order
	local sel_idx = {}
	for idx, action in ipairs(script.actions) do
		if action.selected then
			sel_idx[#sel_idx + 1] = idx
		end
	end

	if #sel_idx == 0 then
		print("stretch_to_current_time: no selected points")
		return
	end
	if #sel_idx == 1 then
		print("stretch_to_current_time: only one selected point, nothing to stretch")
		return
	end

	local n = #sel_idx

	-- 2) snapshot ORIGINAL times and positions (must read before mutating)
	local orig_at = {}
	local orig_pos = {}
	local t0 = math.huge
	local t1 = -math.huge
	for i, idx in ipairs(sel_idx) do
		local a = script.actions[idx].at
		orig_at[i] = a
		orig_pos[i] = script.actions[idx].pos
		if a < t0 then t0 = a end
		if a > t1 then t1 = a end
	end

	local T = player.CurrentTime()
	local new_t0, new_t1
	if T >= t0 then
		new_t0, new_t1 = t0, T
	else
		new_t0, new_t1 = T, t1
	end
	print(string.format("stretch_to_current_time: %d selected points, span [%.3f .. %.3f] -> [%.3f .. %.3f]", n, t0, t1, new_t0, new_t1))

	if T == t0 then
		print("stretch_to_current_time: playhead equals the left edge, nothing to stretch")
		return
	end

	local span = t1 - t0
	local fps = player.FPS()

	-- 3) sort the selected points by ORIGINAL time so the curve is bracketed
	local order = {}
	for i = 1, n do order[i] = i end
	table.sort(order, function(a, b) return orig_at[a] < orig_at[b] end)

	local times = {}
	local positions = {}
	for k = 1, n do
		local i = order[k]
		times[k] = orig_at[i]
		positions[k] = orig_pos[i]
	end

	-- 4) compute new times (snapped to the nearest frame) and the position
	--    from the original curve at the equivalent original time; store the
	--    result in a local array (do NOT mutate script.actions in place)
	local new_points = {}
	for i = 1, n do
		local raw
		if T >= t0 then
			-- left edge pinned at t0, right edge moved to T
			raw = t0 + (T - t0) * (orig_at[i] - t0) / span
		else
			-- left edge moved to T, right edge pinned at t1
			raw = T + (t1 - T) * (orig_at[i] - t0) / span
		end
		local new_at = math.floor(raw * fps + 0.5) / fps

		local t_equiv
		if T >= t0 then
			t_equiv = t0 + span * (new_at - t0) / (T - t0)
		else
			t_equiv = t0 + span * (new_at - T) / (t1 - T)
		end
		local new_pos = curve_pos_at(times, positions, t_equiv)
		local new_p = math.floor(new_pos + 0.5)
		if new_p < 0 then new_p = 0 end
		if new_p > 100 then new_p = 100 end

		local idx = sel_idx[i]
		print(string.format("stretch_to_current_time: idx=%d old_at=%.3f new_at=%.3f t_equiv=%.3f old_pos=%d new_pos=%d", idx, orig_at[i], new_at, t_equiv, orig_pos[i], new_p))
		new_points[#new_points + 1] = { at = new_at, pos = new_p }
	end

	-- 5) deduplicate new points by frame index (keep the last occurrence):
	--    the snap to the nearest frame can map two points onto the same frame
	local dedup = {}
	local dedup_by_frame = {}
	for _, pt in ipairs(new_points) do
		local frame = math.floor(pt.at * fps + 0.5)
		if dedup_by_frame[frame] then
			dedup[dedup_by_frame[frame]] = pt
		else
			dedup_by_frame[frame] = #dedup + 1
			dedup[#dedup + 1] = pt
		end
	end
	table.sort(dedup, function(a, b) return a.at < b.at end)
	print(string.format("stretch_to_current_time: %d selected -> %d point(s) after frame dedup", #new_points, #dedup))

	-- 6) remove ALL selected originals AND any (selected or not) action whose
	--    frame collides with a new point's frame
	local new_frames = {}
	for _, pt in ipairs(dedup) do
		new_frames[math.floor(pt.at * fps + 0.5)] = true
	end
	local removed = 0
	for idx, action in ipairs(script.actions) do
		if action.selected or new_frames[math.floor(action.at * fps + 0.5)] then
			script:markForRemoval(idx)
			removed = removed + 1
		end
	end
	script:removeMarked()

	-- 7) add the new (deduplicated) points
	for _, pt in ipairs(dedup) do
		script.actions:add(Action.new(pt.at, pt.pos, false))
	end

	script:commit()
	print(string.format("stretch_to_current_time: removed %d action(s), inserted %d point(s)", removed, #dedup))
end

-- Find the extremum points (local maxima and minima) of a list of points
-- sorted by time. Returns a list of point indices (1-based) where the curve
-- changes direction. The first and last points are always included as
-- boundaries. Points are expected as a list of {at=..., pos=...} tables.
local function find_extrema(points)
	local n = #points
	if n == 0 then
		return {}
	end
	if n == 1 then
		return { 1 }
	end

	local extrema = { 1 }

	-- interior points: a point is an extremum if its position is strictly
	-- greater than both neighbors (max) or strictly less than both (min)
	for i = 2, n - 1 do
		local prev = points[i - 1].pos
		local cur = points[i].pos
		local next = points[i + 1].pos
		if (cur > prev and cur > next) or (cur < prev and cur < next) then
			extrema[#extrema + 1] = i
		end
	end

	extrema[#extrema + 1] = n
	return extrema
end

-- Find the true extremum points (local maxima and minima) of a list of points
-- sorted by time. Returns a list of { at=..., is_peak=... } tables. A point is
-- a true extremum if it is strictly higher than both neighbors (peak) or
-- strictly lower than both neighbors (valley). The leftmost and rightmost
-- points are included if they are higher/lower than their single neighbor.
local function get_true_extrema(points)
	local extrema = {}
	local n = #points


	-- prev_direction -1 - undefined, 0 - pos_up, 1 - pos_down
	local prev_direction = -1
	local cur_direction = -1

	local prev_action = nil
	local start_idx = 1
	local end_idx = 1

    for i = 2, n do
		local is_peak = nil

		if points[i].pos > points[i-1].pos then
			cur_direction = 0
		elseif points[i].pos < points[i-1].pos then
			cur_direction = 1
		end

		if cur_direction ~= -1 and prev_direction ~= -1 and cur_direction ~= prev_direction then
			end_idx = i - 1

			if cur_direction == 1 then
				is_peak = true
			else
				is_peak = false
			end

			if start_idx == 1 then
				extrema[#extrema + 1] = { at = points[1].at, pos = points[1].pos, is_peak = not is_peak }
			end

			extrema[#extrema + 1] = { at = points[end_idx].at, pos = points[end_idx].pos, is_peak = is_peak }

			start_idx = end_idx
		end

		if i == n then
			if #extrema == 0 then
				if cur_direction == 1 then
					extrema[#extrema + 1] = { at = points[1].at, pos = points[1].pos, is_peak = true }
				else
					extrema[#extrema + 1] = { at = points[1].at, pos = points[1].pos, is_peak = false }
				end
			end

			if cur_direction == 1 then
				extrema[#extrema + 1] = { at = points[n].at, pos = points[n].pos, is_peak = false }
			else
				extrema[#extrema + 1] = { at = points[n].at, pos = points[n].pos, is_peak = true }
			end
		end

		prev_direction = cur_direction
    end

	return extrema
end

-- Find traget points that contain only peaks and no any intermediate points
local function get_target_valid_extrema(points)
	local extrema = {}
	local n = #points

	-- prev_direction -1 - undefined, 0 - pos_up, 1 - pos_down
	local prev_direction = -1
	local cur_direction = -1

	local prev_action = nil
	local start_idx = 1
	local end_idx = 1

    for i = 2, n do
		print(string.format("get_target_valid_extrema: 1: i=%d n=%d points[i].at=%g points[i].pos=%g points[i-1].at=%g points[i-1].pos=%g", i, n, points[i].at, points[i].pos, points[i-1].at, points[i-1].pos))

		local is_peak = nil

		if points[i].pos > points[i-1].pos then
			cur_direction = 0
		elseif points[i].pos < points[i-1].pos then
			cur_direction = 1
		else
			break
		end

		print(string.format("get_target_valid_extrema: 2: i=%d cur_direction=%d", i, cur_direction))

		if cur_direction == prev_direction or points[i].pos == points[i-1].pos then
			break
		end

		print(string.format("get_target_valid_extrema: 3"))

		if cur_direction ~= -1 and prev_direction ~= -1 then
			end_idx = i - 1

			if cur_direction == 1 then
				is_peak = true
			else
				is_peak = false
			end

			if start_idx == 1 then
				extrema[#extrema + 1] = { at = points[1].at, pos = points[1].pos, is_peak = not is_peak }
			end

			extrema[#extrema + 1] = { at = points[end_idx].at, pos = points[end_idx].pos, is_peak = is_peak }

			start_idx = end_idx
		end

		if i == n then
			print(string.format("get_target_valid_extrema: 4: i=%d == n", i))
			if #extrema == 0 then
				if cur_direction == 1 then
					extrema[#extrema + 1] = { at = points[1].at, pos = points[1].pos, is_peak = true }
				else
					extrema[#extrema + 1] = { at = points[1].at, pos = points[1].pos, is_peak = false }
				end
			end

			if cur_direction == 1 then
				extrema[#extrema + 1] = { at = points[n].at, pos = points[n].pos, is_peak = false }
			else
				extrema[#extrema + 1] = { at = points[n].at, pos = points[n].pos, is_peak = true }
			end
		end

		prev_direction = cur_direction
    end

	return extrema
end

-- Insert the copied buffer at the current video time, stretching each
-- monotonic segment (rising/falling) of the copied buffer to match the
-- corresponding segment of the target graph (all actions of the active
-- script at or after the current time).
--
-- Rules:
-- * If the copied curve has exactly 1 monotonic segment, the entire curve
--   is stretched onto the first two target points (the one at/after T and
--   the next one), regardless of whether they are extrema.
-- * If the copied curve has more than 1 monotonic segment, the direction of
--   its first segment must match the direction of the target curve at the
--   insertion point; otherwise an error is printed and nothing is inserted.
-- * Pinned points (matching extrema + boundary points) keep their TARGET
--   time AND position unchanged. Intermediate points are stretched
--   horizontally (time mapping) AND vertically (position rescaled from the
--   copied segment's [start,end] range to the target segment's [start,end]).
-- * Times are snapped to the nearest video frame. Positions are rounded to
--   the nearest integer (0-100).
function insert_with_stretch()
	print("insert_with_stretch: Starting ...")

	local script = ofs.Script(ofs.ActiveIdx())

	-- 1) get the copied buffer (Ctrl+C)
	local copied = ofs.Clipboard()
	if not copied or #copied.actions < 2 then
		print("insert_with_stretch: copied buffer is empty or has fewer than 2 points")
		return
	end

	-- 2) extract copied points, sorted by time
	local copied_points = {}
	for _, action in ipairs(copied.actions) do
		copied_points[#copied_points + 1] = { at = action.at, pos = action.pos }
	end
	table.sort(copied_points, function(a, b) return a.at < b.at end)
	local n_copied = #copied_points

	-- 3) get copied true extrema (handles flat runs correctly via cur_direction persistence)
	local copied_true_extrema = get_true_extrema(copied_points)
	local n_copied_segments = #copied_true_extrema - 1
	-- first direction: from copied_points[1] to the second extremum
	local copied_first_dir = -1
	if #copied_true_extrema >= 2 then
		if copied_true_extrema[2].pos > copied_points[1].pos then
			copied_first_dir = 0
		elseif copied_true_extrema[2].pos < copied_points[1].pos then
			copied_first_dir = 1
		end
	end
	print(string.format("insert_with_stretch: copied segments=%d first_dir=%d", n_copied_segments, copied_first_dir))

	-- 4) get target points: all actions of the active script at or after
	--    the current time, sorted by time
	local T = player.CurrentTime()
	local fps = player.FPS()
	local fd = 0
	if fps > 0 then
		fd = 1.0 / fps
	end
	T = T - (fd / 3.0)
	local target_points = {}
	for _, action in ipairs(script.actions) do
		if action.at >= T then
			print(string.format("insert_with_stretch: T target_points[%g].at=%g target_points[%g].pos=%g", #target_points + 1, action.at, #target_points + 1, action.pos))
			target_points[#target_points + 1] = { at = action.at, pos = action.pos }
			if #target_points >= n_copied_segments + 1 then
				break
			end
		end
	end
	if (#target_points > 0) then
		T = target_points[1].at
	end
	print(string.format("insert_with_stretch: T: %g player.CurrentTime: %g", T, player.CurrentTime()))
	table.sort(target_points, function(a, b) return a.at < b.at end)
	local n_target = #target_points

	if n_target < 2 then
		print("insert_with_stretch: fewer than 2 target points at or after current time")
		return
	end
	print(string.format("insert_with_stretch: #target_points == %d", n_target))

	-- 5) determine target first direction (from target_points[1] to target_points[2])
	local target_first_dir
	if target_points[2].pos > target_points[1].pos then
		target_first_dir = 0
	elseif target_points[2].pos < target_points[1].pos then
		target_first_dir = 1
	else
		target_first_dir = -1 -- flat, direction unknown
	end

	-- 6) build anchors: list of {x=copied_time, y=new_time, cp=copied_pos, tp=target_pos}
	--    For single-segment case: stretch entire copied curve onto target_points[1..2]
	--    For multi-segment case: match extrema by type, pin them to target positions
	local anchors = {}

	if n_copied_segments <= 1 then
		-- Special case: 1 monotonic segment (possibly with flat runs).
		-- Stretch the whole curve onto the first two target points.
		-- The target points may or may not be extrema.
		local t1_at, t1_pos = target_points[1].at, target_points[1].pos
		local t2_at, t2_pos = target_points[2].at, target_points[2].pos

		anchors[1] = { x = copied_points[1].at, y = t1_at, cp = copied_points[1].pos, tp = t1_pos }
		anchors[2] = { x = copied_points[n_copied].at, y = t2_at, cp = copied_points[n_copied].pos, tp = t2_pos }

		print(string.format("insert_with_stretch: single-segment case, target span [%.3f, %.3f] pos [%d, %d]",
			t1_at, t2_at, t1_pos, t2_pos))

	else
		-- Multi-segment case: direction check
		if copied_first_dir ~= -1 and target_first_dir ~= -1 then
			if copied_first_dir ~= target_first_dir then
				print(string.format(
					"insert_with_stretch: ERROR - direction mismatch: copied first dir=%s, target first dir=%s. Aborting.",
					copied_first_dir == 0 and "up" or "down",
					target_first_dir == 0 and "up" or "down"))
				return
			end
		end

		-- Target valid extrema for matching
		local target_true_extrema = get_target_valid_extrema(target_points)
		local n_target_true = #target_true_extrema
		print(string.format("insert_with_stretch: #copied_true_extrema=%d #target_true_extrema=%d", #copied_true_extrema, n_target_true))

		if n_target_true < 1 then
			print("insert_with_stretch: target has no true extrema")
			return
		end

		-- Anchor 0: copied leftmost point at playhead T
		anchors[1] = { x = copied_points[1].at, y = T, cp = copied_points[1].pos, tp = target_points[1].pos }

		-- Match copied extrema to target extrema by type (peak->peak, valley->valley)
		local target_idx = 1
		local map_count = 0
		for i = 1, #copied_true_extrema do
			local ce = copied_true_extrema[i]
			-- skip the copied graph's leftmost point (already anchored)
			local is_leftmost = (i == 1 and ce.at == copied_points[1].at)
			if not is_leftmost then
				local found = false
				for j = target_idx, n_target_true do
					if target_true_extrema[j].is_peak == ce.is_peak then
						if ce.at > anchors[#anchors].x then
							anchors[#anchors + 1] = {
								x = ce.at,
								y = target_true_extrema[j].at,
								cp = ce.pos,
								tp = target_true_extrema[j].pos
							}
							map_count = map_count + 1
						end
						target_idx = j + 1
						found = true
						break
					end
				end
				if not found then
					break
				end
			end
		end

		print(string.format("insert_with_stretch: anchor_count=%d map_count=%d", #anchors, map_count))
		for i = 1, #anchors do
			print(string.format("insert_with_stretch: anchor %d: x=%.3f y=%.3f cp=%d tp=%d",
				i, anchors[i].x, anchors[i].y, anchors[i].cp, anchors[i].tp))
		end
	end

	local anchor_count = #anchors

	-- 7) compute new points with horizontal stretch + vertical rescale
	local new_points = {}
	local last_anchor_x = anchors[anchor_count].x

	local times = {}
	local positions = {}
	for i = 1, n_copied do
		times[i] = copied_points[i].at
		positions[i] = copied_points[i].pos
	end

	for i = 1, n_copied do
		local t = copied_points[i].at
		if t > last_anchor_x then
			break -- drop copied points beyond the last mapped anchor
		end

		local new_at
		local new_pos

		if t <= anchors[1].x then
			-- leftmost anchor: use target position directly
			new_at = anchors[1].y
			new_pos = anchors[1].tp
		elseif t >= last_anchor_x then
			-- rightmost anchor: use target position directly
			new_at = anchors[anchor_count].y
			new_pos = anchors[anchor_count].tp
		else
			-- find which segment this point falls in
			local seg = anchor_count - 1
			for a = 1, anchor_count - 1 do
				if t >= anchors[a].x and t <= anchors[a + 1].x then
					seg = a
					break
				end
			end

			local ax = anchors[seg].x
			local bx = anchors[seg + 1].x
			local ay = anchors[seg].y
			local by = anchors[seg + 1].y
			local cp_a = anchors[seg].cp
			local cp_b = anchors[seg + 1].cp
			local tp_a = anchors[seg].tp
			local tp_b = anchors[seg + 1].tp

			local span_x = bx - ax
			local span_y = by - ay

			if span_x > 0 then
				-- horizontal: linear time mapping
				local raw_t = ay + span_y * (t - ax) / span_x
				new_at = math.floor(raw_t * fps + 0.5) / fps

				-- inverse: what copied time does this new_at correspond to?
				local t_equiv
				if span_y > 0 then
					t_equiv = ax + span_x * (new_at - ay) / span_y
				else
					t_equiv = ax
				end

				-- vertical: resample from copied curve, then rescale to target range
				local p_raw = curve_pos_at(times, positions, t_equiv)
				local span_c = cp_b - cp_a
				if span_c ~= 0 then
					new_pos = tp_a + (p_raw - cp_a) * (tp_b - tp_a) / span_c
				else
					-- flat copied segment: map to target start
					new_pos = tp_a
				end
			else
				-- degenerate time span
				new_at = ay
				new_pos = tp_a
			end
		end

		new_pos = math.floor(new_pos + 0.5)
		if new_pos < 0 then new_pos = 0 end
		if new_pos > 100 then new_pos = 100 end

		new_points[#new_points + 1] = { at = new_at, pos = new_pos }
	end

	-- 8) deduplicate new points by frame index (keep the last occurrence)
	local dedup = {}
	local dedup_by_frame = {}
	for _, pt in ipairs(new_points) do
		local frame = math.floor(pt.at * fps + 0.5)
		if dedup_by_frame[frame] then
			dedup[dedup_by_frame[frame]] = pt
		else
			dedup_by_frame[frame] = #dedup + 1
			dedup[#dedup + 1] = pt
		end
	end
	table.sort(dedup, function(a, b) return a.at < b.at end)

	-- 9) replace the target's points in the mapped region with the new
	--    points. Remove all target actions in [T, last anchor time] and
	--    any that collide with new point frames.
	local range_start = T
	local range_end = anchors[anchor_count].y

	local new_frames = {}
	for _, pt in ipairs(dedup) do
		local frame = math.floor(pt.at * fps + 0.5)
		new_frames[frame] = pt
	end

	for idx, action in ipairs(script.actions) do
		local in_range = (action.at >= range_start and action.at <= range_end)
		local frame = math.floor(action.at * fps + 0.5)
		local collides = new_frames[frame] ~= nil
		if in_range or collides then
			script:markForRemoval(idx)
		end
	end
	script:removeMarked()

	-- add the new points
	for _, pt in ipairs(dedup) do
		script.actions:add(Action.new(pt.at, pt.pos, false))
	end

	script:commit()
	print("insert_with_stretch: done, " .. tostring(#dedup) .. " points inserted")
end
