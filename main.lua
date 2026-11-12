-- ═══════════════════════════════════════════════════════════
-- Hismith Funscript Generator - OFS Extension
--
-- Generates funscript points from a video (pelvis point tracking)
-- and inserts them into the current OpenFunscripter script starting
-- at the current player position.
--
-- Extension folder layout (%APPDATA%\OFS\OFS3_data\extensions\Hismith Funscript Generator):
--   Hismith Funscript Generator\
--   ├── main.lua            (this file)
--   ├── funscript_gen.py    (Python script, lives next to this file)
--   └── 3rdParty/
--       └── rxi/
--           └── json.lua    (JSON library by rxi - needed to parse the result)
--
-- Requirements:
--   • Python in PATH:  python funscript_gen.py ...
--   • pip install opencv-python mediapipe numpy scipy tqdm
--
-- Usage:
--   1. Open a video in OFS
--   2. Seek the player to the generation start
--   3. Extensions > Hismith Funscript Generator > open the window
--   4. Pick tracker, motion mode, normalization
--   5. Press [Generate from here]
--   6. Wait for completion (identify + f/b calibration windows
--      open first; mark them and confirm)
--   7. The points appear on the timeline (selected, editable)
-- ═══════════════════════════════════════════════════════════

-- json.lua lives in 3rdParty/rxi/ (vendored from https://github.com/rxi/json.lua,
-- MIT license, see the header inside the file). OFS adds <extdir>/?.lua to
-- package.path (OFS_LuaExtension.cpp), so the dotted module path resolves to
-- <extdir>/3rdParty/rxi/json.lua.
json = require "3rdParty.rxi.json"

-- ── globals ──
procHandle = nil
status = "Idle"
lastError = ""
addedCount = 0

-- settings (survive gui redraws)
trackerIdx = 3    -- 1=yolo, 2=mediapipe, 3..6=sot-*, 7=groundingdino, 8=ostrack
prevTrackerIdx = trackerIdx
yoloModelIdx = 15 -- 1-5=yolov8 n..x, 6-10=yolo11 n..x, 11-15=yolo26 n..x (default 15=yolo26x-pose)
dinoModelIdx = 1  -- 1=swinb (default, Swin-B 56.7 AP), 2=swint (Swin-T 48.4 AP)
-- GroundingDINO target list - INDEPENDENT of the pose-tracker "point" combo
-- (by design: DINO targets are free-text prompts, not keypoint zones).
-- The list is queried from Python (funscript_gen.py --list dino, the single
-- source of truth = core/detectors/groundingdino.py DINO_TARGETS) and
-- CACHED here (ensureDinoTargets) - NEVER inside the per-frame gui() draw
-- (that spawns a python process each time; the lazy-scan pattern is the
-- same as ensureFamilyModels above).
dinoTargets = {}
dinoTargetIdx = 1     -- 1 = DEFAULT_DINO_TARGET (printed first by --list dino)
lastDinoScan = nil    -- cached result of ensureDinoTargets (nil = not scanned)
dinoCustomPrompt = "" -- non-empty = overrides the target's built-in phrase
-- two SEPARATE ONNX weight pickers - each is shown ONLY for the tracker that
-- consumes the folder (exactly like the YOLO combo shows only when
-- trackerIdx==1 and the DINO combo only when trackerIdx==7):
--   * ostrackModel / ostrackModelIdx -> models/ostrack/*.onnx   (OSTrack weights, tracker idx 8)
--   * vitModels    / vitModelIdx    -> models/vittrack/*.onnx  (sot-vit TrackerVit weights, tracker idx 5)
-- sot-mil trains its classifier online, csrt/kcf are self-contained —
-- NEITHER combo applies to them (the pickers only forward when the tracker
-- consumes them, i.e. tracker==ostrack or tracker==sot-vit).
-- Both lists are scanned LAZILY (see ensureFamilyModels) - never inside the
-- per-frame gui() draw loop (a blocking `dir /b` there is what froze the
-- GUI in earlier versions).
ostrackModels = {}
ostrackModelIdx = 1 -- 1 = auto (first in models/ostrack/)
vitModels = {}
vitModelIdx = 1     -- 1 = auto (first in models/vittrack/)
-- Grounded SAM2 (gsam2, tracker idx 9): checkpoint list comes from
-- `funscript_gen.py --list sam2` (single source of truth =
-- core/detectors/models.py SAM2_VARIANTS, default printed FIRST so combo
-- index 1 = it), LAZY+CACHED like dinoTargets (see ensureSam2Models).
-- The point-p axis mode is a small fixed combo (cos / principal / bbox).
sam2Models = {}
sam2ModelIdx = 1      -- 1 = default variant (printed first by --list sam2)
lastSam2Scan = nil    -- cached result of ensureSam2Models (nil = not scanned)
sam2AxisIdx = 1       -- 1=cos (default), 2=principal, 3=bbox
sam2FramesIdx = 1     -- 1=lazy (default, constant RAM), 2=jpeg (RAM grows)
sam2Ema = "0.5"       -- --sam2-ema default; string (ofs.Input) - forwarded
-- only when tonumber() is a valid 0.0..1.0 value
pointIdx = 1          -- 1=mid-hips (default), 2=left-hip, 3=right-hip, 4=thigh, 5=torso, 6=shoulders
flowIdx = 1           -- 1=off (default), 2=pure, 3=fusion
useCalibrate = true   -- interactive f/b calibration window (default on)
autoPartners = false  -- skip the identify window (auto: person1=active, person2=passive)
noSceneRoi = false    -- SOT only: skip the scene-ROI setup window (full frame, identity transform)
useVlm = false
-- ── LLM (Ollama) generation mode: tracker idx 10 ──
-- Single source of truth for the LLM CLI defaults: core/config.py
-- LLM_DEFAULT_* (0.5s / 200000 / 32000 / 2048 / 1920).  The GUI
-- mirrors the same defaults (see funscript_gen.py CLI help for each
-- knob).  The Ollama model name is NOT a fixed list - it comes from
-- `funscript_gen.py --list ollama` (a LAZY+CACHED wrapper over
-- `ollama list`), same pattern as `ensureDinoTargets` and
-- `ensureFamilyModels` above.  See plans/llm-ollama-generation.md.
ollamaModels = {}
ollamaModelIdx = 1    -- 1 = first in the list (default)
lastOllamaScan = nil  -- cached result of ensureOllamaModels
llmClipS = "0.5"      -- --llm-clip-s default (string: ofs.Input)
llmNumCtx = "200000"  -- --num-ctx default
llmNumPredict = "32000"
llmNumBatch = "2048"
llmThumb = "1920"
llmPrompt = ""        -- --llm-prompt (optional; empty = default)
llmStop = ""          -- --llm-stop (MM:SS; empty = end of video)
minIntervalMs = 42
latencyMs = 0
-- motion interpretation & normalization
motionIdx = 4    -- 1=straight, 2=arc, 3=parabola, 4=complex (raw path, default)
normalizeIdx = 1 -- 1=per-stroke (each=100%), 2=global-max
pythonCmd = "python.exe"

-- ── settings persistence (hfg_settings.json in the extension dir) ──
-- The OFS extension API has no built-in settings store, so every GUI
-- setting that should survive an OFS restart is saved to a small JSON
-- file next to main.lua (ofs.ExtensionDir()).  loadSettings() runs once
-- in init(); saveSettings() runs whenever a GUI control changes.
SETTINGS_FILE = "hfg_settings.json"
SETTINGS_FIELDS = {
    -- tracker & model selection
    trackerIdx = "num", yoloModelIdx = "num", dinoModelIdx = "num",
    dinoTargetIdx = "num", ostrackModelIdx = "num", vitModelIdx = "num",
    sam2ModelIdx = "num", sam2AxisIdx = "num", sam2FramesIdx = "num",
    ollamaModelIdx = "num",
    -- tracked point & smoothing
    pointIdx = "num", flowIdx = "num",
    -- interactive windows
    useCalibrate = "bool", autoPartners = "bool", noSceneRoi = "bool",
    useVlm = "bool",
    -- LLM (Ollama) generation mode
    llmClipS = "str", llmNumCtx = "str", llmNumPredict = "str",
    llmNumBatch = "str", llmThumb = "str", llmPrompt = "str", llmStop = "str",
    sam2Ema = "str", dinoCustomPrompt = "str",
    -- timing
    minIntervalMs = "num", latencyMs = "num",
    -- motion interpretation & normalization
    motionIdx = "num", normalizeIdx = "num",
    -- python interpreter
    pythonCmd = "str",
}

-- ── utilities ──

function fileExists(path)
    local f = io.open(path, "r")
    if f then f:close() end
    return f ~= nil
end

function settingsPath()
    return ofs.ExtensionDir() .. "\\" .. SETTINGS_FILE
end

function loadSettings()
    local ok, err = pcall(function()
        local f = io.open(settingsPath(), "r")
        if not f then return end
        local content = f:read("*a")
        f:close()
        if not content or #content == 0 then return end
        local data = json.decode(content)
        if type(data) ~= "table" then return end
        local changed = false
        for k, typ in pairs(SETTINGS_FIELDS) do
            local v = data[k]
            if v ~= nil then
                if (typ == "num" and type(v) == "number")
                    or (typ == "str" and type(v) == "string")
                    or (typ == "bool" and type(v) == "boolean") then
                    _G[k] = v
                    changed = true
                end
            end
        end
        if changed then
            print("[HF] settings loaded from " .. SETTINGS_FILE)
        end
    end)
    if not ok then
        print("[HF] loadSettings error: " .. tostring(err))
    end
end

function saveSettings()
    local ok, err = pcall(function()
        local data = {}
        for k, typ in pairs(SETTINGS_FIELDS) do
            local v = _G[k]
            if v ~= nil then
                if (typ == "num" and type(v) == "number")
                    or (typ == "str" and type(v) == "string")
                    or (typ == "bool" and type(v) == "boolean") then
                    data[k] = v
                end
            end
        end
        local f = io.open(settingsPath(), "w")
        if not f then error("cannot open " .. settingsPath() .. " for writing") end
        f:write(json.encode(data))
        f:close()
    end)
    if not ok then
        print("[HF] saveSettings error: " .. tostring(err))
    end
end

-- ── per-family ONNX list (one folder at a time - ostrack or vittrack) ──
-- listOnnxIn(extDir, subdir) - one blocking "dir /b" of models/<subdir>/*.onnx
-- (sorted).  Returns the list of FILENAMES (bare, no path).  subdir is
-- "ostrack" or "vittrack" (see core/config.py MODEL_DIRS - the single source
-- of truth for these names).  Do NOT call this from gui()'s per-frame draw:
-- it spawns a cmd process each time, which is what froze the extension.

local function listOnnxIn(extDir, subdir)
    local out = {}
    local ok, err = pcall(function()
        local dir = extDir:gsub("/", "\\") .. "\\models\\" .. subdir .. "\\*.onnx"
        local cmd = 'dir /b "' .. dir .. '" 2>nul'
        print("[HF] listOnnxIn cmd: " .. cmd)
        local pipe = io.popen(cmd)
        if not pipe then error("io.popen unavailable") end
        for line in pipe:lines() do
            -- `dir /b` prints the pattern itself when NOTHING matched;
            -- skip it plus blank/short lines
            local t = line:match("^%s*(.-)%s*$") or ""
            if #t > 3 and t ~= "*.onnx" then
                table.insert(out, t)
            end
        end
        pipe:close()
    end)
    table.sort(out)
    return out, (ok and true or false), tostring(err)
end

-- ensureFamilyModels(extDir, family) - the ONLY place the ONNX weight
-- folders are read from the GUI (family = "ostrack" or "vittrack").

local function ensureFamilyModels(extDir, family)
    local list = (family == "vittrack") and vitModels or ostrackModels
    local fresh, ok, err = listOnnxIn(extDir, family)
    if family == "vittrack" then
        vitModels = fresh
        lastVitScan = fresh
    else
        ostrackModels = fresh
        lastOstrackScan = fresh
    end
    local tail = ok and "" or (" (popen error: " .. err .. ")")
    print(("[HF] %s ONNX list refreshed: %d file(s)" .. tail)
        :format(family, #fresh))
    return (family == "vittrack") and vitModels or ostrackModels
end

-- ensureDinoTargets(extDir, pythonCmd) - query the GroundingDINO target
-- list from Python (`funscript_gen.py --list dino`: the single source of
-- truth is core/detectors/groundingdino.py DINO_TARGETS, one name per
-- line; the default target is printed FIRST so combo index 1 = it).
-- LAZY + CACHED (lastDinoScan): exactly one python process per GUI
-- session; a re-scan only happens when the combo index goes out of range
-- (never on idle frames).  On any failure the placeholder list is kept so
-- the GUI stays functional (the CLI still works with its own defaults).

local function ensureDinoTargets(extDir, pythonCmd)
    if lastDinoScan then
        dinoTargets = lastDinoScan
        return dinoTargets
    end
    local out = {}
    local py = extDir:gsub("/", "\\") .. "\\funscript_gen.py"
    local ok = pcall(function()
        local cmd = '""' .. pythonCmd .. '" "' .. py .. '" --list dino""'
        print("[HF] ensureDinoTargets cmd: " .. cmd)
        local pipe = io.popen(cmd)
        if not pipe then error("io.popen unavailable") end
        for line in pipe:lines() do
            print("[HF] line: " .. line)
            local t = line:match("^%s*(.-)%s*$") or ""
            if #t > 1 then
                table.insert(out, t)
            end
        end
        pipe:close()
    end)
    if not ok or #out == 0 then
        out = { "(dino target list unavailable - using built-in default)" }
        print("[HF] dino target list query failed or empty; using fallback entry")
    end
    lastDinoScan = out
    dinoTargets = out
    print(("[HF] dino target list loaded: %d target(s)"):format(#out))
    return out
end


-- ensureOllamaModels(extDir, pythonCmd) - query the Ollama model list
-- from a LAZY call to `funscript_gen.py --list ollama` (a thin wrapper
-- over `ollama list` via the Python client).  Cached (lastOllamaScan)
-- like dinoTargets so the per-frame draw loop stays non-blocking.
-- If the Python client / Ollama server is unavailable, shows a
-- placeholder the user can still select (the CLI will fail with a
-- clear error when the tracker runs).
local function ensureOllamaModels(extDir, pythonCmd)
    if lastOllamaScan then
        ollamaModels = lastOllamaScan
        return ollamaModels
    end
    local py = extDir:gsub("/", "\\") .. "\\funscript_gen.py"
    -- Mirror the proven ensureDinoTargets quoting EXACTLY (no `2>nul` suffix;
    -- that redirection combined with the closing `""` in io.popen leaked a
    -- trailing `;` into the argv token on Windows, making the Python CLI
    -- see `--list ollama;` and refuse to print the list).
    local cmd = '""' .. pythonCmd .. '" "' .. py .. '" --list ollama""'
    local out = {}
    local ok, err = pcall(function()
        local pipe = io.popen(cmd)
        if pipe then
            for line in pipe:lines() do
                if line and #line > 0 then
                    table.insert(out, line)
                end
            end
            pipe:close()
        end
    end, function(e)
        err = e
        return e
    end)
    if (not ok) or (#out == 0) then
        table.insert(out, "(no Ollama models found - start `ollama serve` and `ollama pull <name>` first; you can still type a name below and press Generate)")
    end
    lastOllamaScan = out
    ollamaModels = out
    print(("[HF] ollama model list loaded: %d model(s)"):format(#out))
    return out
end

-- ensureSam2Models(extDir, pythonCmd) - query the SAM 2 checkpoint picker
-- list from Python (`funscript_gen.py --list sam2`: the single source of
-- truth is core/detectors/models.py SAM2_VARIANTS, default FIRST).
-- LAZY + CACHED (lastSam2Scan): one python process per GUI session; the
-- list is a fixed set of official variants, so it never needs re-scanning
-- except when the combo index goes out of range.  On failure a placeholder
-- is kept so the GUI stays functional (CLI still uses its own default).
local function ensureSam2Models(extDir, pythonCmd)
    if lastSam2Scan then
        sam2Models = lastSam2Scan
        return sam2Models
    end
    local out = {}
    local py = extDir:gsub("/", "\\") .. "\\funscript_gen.py"
    local ok = pcall(function()
        local dq = '"'
        -- mirrors the proven ensureDinoTargets command quoting exactly:
        --   ""<pythonCmd> "<py>" --list sam2""
        local cmd = dq .. dq .. pythonCmd .. dq .. " " .. dq .. py .. dq
            .. " --list sam2" .. dq .. dq
        print("[HF] ensureSam2Models cmd: " .. cmd)
        local pipe = io.popen(cmd)
        if not pipe then error("io.popen unavailable") end
        for line in pipe:lines() do
            local t = line:match("^%s*(.-)%s*$") or ""
            if #t > 1 then
                table.insert(out, t)
            end
        end
        pipe:close()
    end)
    if not ok or #out == 0 then
        out = { "sam2.1-hiera-base-plus",
            "(sam2 model list unavailable - using default)" }
        print("[HF] sam2 model list query failed or empty; using fallback entry")
    end
    lastSam2Scan = out
    sam2Models = out
    print(("[HF] sam2 model list loaded: %d variant(s)"):format(#out))
    return out
end

-- ── start generation ──

function startGeneration()
    if procHandle then
        status = "Already running"
        return
    end

    local video = player.CurrentVideo()

    -- If the player returned nothing, enable the fallback path
    if not video or video == "" then
        local script = ofs.Script(0)
        if script and script.Metadata and script.Metadata.Video then
            video = script.Metadata.Video
        end
    end

    -- Final check: if the metadata is also empty, then no video is really open
    if not video or video == "" then
        status = "ERROR: open a video first!"
        return
    end

    local scriptIdx = ofs.ActiveIdx()
    local script = ofs.Script(scriptIdx)
    local startTime = player.CurrentTime()
    local endTime = player.Duration() -- until the end of the video (stop = q/space in the preview window)

    local extDir = ofs.ExtensionDir()
    local pyScript = extDir .. "/funscript_gen.py"
    local outFile = extDir .. "/ofs_result.json"

    if not fileExists(pyScript) then
        status = "ERROR: funscript_gen.py not found in " .. extDir
        return
    end

    -- ─── existing points on the timeline ───
    -- The streaming loop stops (ALWAYS) when it reaches the first pre-existing
    -- action at or after t0 (requirement #2) and the post-pass rescales the
    -- boundary stroke(s) so the new track lands EXACTLY on those anchor values
    -- (requirement #3).  Both need: (a) the full list of existing action
    -- times (for the stop condition, compared 3-decimals-against-absolute-
    -- video-seconds) and (b) the anchor BEGIN/END values (nearest existing
    -- points on either side of [t0, t0] -- "before" inclusive, "after"
    -- exclusive -- the last one before t0 is the boundary at the start and
    -- the first one after t0 is the boundary at the end).
    -- Dumped to a plain-text file (one "t" per line, 3 decimals) so the
    -- Python side can read it without any JSON dependency.
    local t0 = startTime
    local existingFile = extDir:gsub("/", "\\") .. "\\hf_existing_points.txt"
    -- Dump every existing action (t, pos) one per line. The Python side
    -- reads it and uses:
    --   * the times: to STOP the streaming at the first pre-existing
    --     point at-or-after start_s (requirement #2);
    --   * the pos of the LAST point BEFORE start_s (anchor_begin_pos):
    --     to rescale the first stroke(s) so the new segment starts at the
    --     user's existing value (requirement #3);
    --   * the pos of the FIRST point AT-OR-AFTER start_s (anchor_end_pos):
    --     to rescale the last stroke(s) so the new segment lands EXACTLY on
    --     the user's existing value (requirement #3).
    -- Both anchor positions are OPTIONAL (nil if there is no existing
    -- point on that side of start_s) — the post-pass then leaves the
    -- affected stroke unrescaled (legacy behaviour).
    local fileWritten = false
    do
        local f = io.open(existingFile, "w")
        if f then
            local n = 0
            for _, a in ipairs(script.actions) do
                local t = tonumber(a.at)
                local p = tonumber(a.pos)
                if t and t > 0.0005 and p then
                    f:write(string.format("%.3f %.3f\n", t, p))
                    n = n + 1
                end
            end
            f:close()
            fileWritten = (n > 0)
        end
    end
    print(string.format(
        "[HF] existing points file: %s (wrote=%s, t0=%.3f)",
        existingFile,
        fileWritten and "yes" or "no (skipping --existing-points-file)",
        startTime))

    -- clear the previous result
    local of = io.open(outFile, "w")
    if of then of:close() end

    local trackers = { "yolo", "mediapipe", "sot-csrt", "sot-kcf", "sot-vit", "sot-mil", "groundingdino", "ostrack",
        "gsam2", "ollama" }
    -- note: point is only forwarded for yolo/mediapipe (the POSE-tracker
    -- body zone); gsam2 / ostrack / sot-* use the ROI the user draws, so
    -- the --point combo is intentionally hidden for them (and not sent).
    local tracker = trackers[trackerIdx] or "sot-csrt"
    local yoloModels = { "yolov8n-pose", "yolov8s-pose", "yolov8m-pose", "yolov8l-pose", "yolov8x-pose",
        "yolo11n-pose", "yolo11s-pose", "yolo11m-pose", "yolo11l-pose", "yolo11x-pose",
        "yolo26n-pose", "yolo26s-pose", "yolo26m-pose", "yolo26l-pose", "yolo26x-pose" }
    local points = { "mid-hips", "left-hip", "right-hip", "thigh", "torso", "shoulders", "head", "ankle" }
    local flows = { "off", "pure", "fusion" }
    local motions = { "straight", "arc", "parabola", "complex" }
    local normalizes = { "per-stroke", "global-max" }

    local args = {
        pyScript,
        video,
        "--start", string.format("%.3f", startTime),
        "--tracker", tracker,
        "--motion", motions[motionIdx] or "complex",
        "--normalize", normalizes[normalizeIdx] or "per-stroke",
        "--flow", flows[flowIdx] or "off",
        "--output", outFile,
        "--min-interval-ms", tostring(minIntervalMs),
        "--latency-ms", tostring(latencyMs),
    }
    -- existing-points plumbing (requirements #2 and #3): pass the dump file
    -- ONLY if at least one action was written to it (a 0-line file would
    -- still be valid but is equivalent to the flag being absent, and we
    -- want to keep the CLI noise minimal).
    if fileWritten then
        table.insert(args, "--existing-points-file")
        table.insert(args, existingFile)
    end
    -- --point (the POSE-tracker body zone, its list = core/config.TARGETS)
    -- is only valid for yolo / mediapipe: groundingdino has its OWN target
    -- list (--dino-target, queried from Python) and the SOT/ostrack
    -- trackers pick the point by drawing a ROI -- both would reject a
    -- dino-style name, so the pair is simply not sent for them.
    if tracker == "yolo" or tracker == "mediapipe" then
        table.insert(args, "--point")
        table.insert(args, points[pointIdx] or "mid-hips")
    end
    if tracker == "yolo" then
        table.insert(args, "--yolo-model")
        table.insert(args, yoloModels[yoloModelIdx])
    end
    if tracker == "groundingdino" then
        table.insert(args, "--dino-model")
        table.insert(args, (dinoModelIdx == 2) and "swint" or "swinb")
        -- DINO target (its OWN list, independent of points[]/--point above)
        local d = (dinoTargets and dinoTargetIdx and dinoTargets[dinoTargetIdx]) or "pelvis"
        table.insert(args, "--dino-target")
        table.insert(args, d)
        -- optional custom prompt overriding the target's built-in phrase
        if dinoCustomPrompt and dinoCustomPrompt:match("%S") then
            table.insert(args, "--dino-prompt")
            table.insert(args, dinoCustomPrompt)
        end
    end
    if tracker == "ostrack" then
        -- OSTrack ONNX picker: 1 = auto (first in models/ostrack/),
        --                      2+(N-1) = the N-th file from models/ostrack/
        if ostrackModelIdx and ostrackModelIdx > 1 then
            local pick = tostring(ostrackModels[ostrackModelIdx - 1] or "")
            if pick ~= "" then
                table.insert(args, "--ostrack-model")
                table.insert(args, pick)
            end
        end
    end
    if tracker == "sot-vit" then
        -- sot-vit ONNX picker: 1 = auto (auto-download of vitTracker.onnx),
        --                      2+(N-1) = the N-th file from models/vittrack/
        if vitModelIdx and vitModelIdx > 1 then
            local pick = tostring(vitModels[vitModelIdx - 1] or "")
            if pick ~= "" then
                table.insert(args, "--sot-model")
                table.insert(args, pick)
            end
        end
    end
    if tracker == "gsam2" then
        -- SAM 2 variant picker (from --list sam2, cached in sam2Models)
        local sam2v = (sam2Models and sam2ModelIdx and sam2Models[sam2ModelIdx]) or "sam2.1-hiera-base-plus"
        table.insert(args, "--sam2-model")
        table.insert(args, sam2v)
        -- frames backend: lazy (default, constant RAM) | jpeg (RAM grows)
        local sam2Frames = { "lazy", "jpeg" }
        local fb = sam2Frames[sam2FramesIdx or 1] or "lazy"
        table.insert(args, "--sam2-frames")
        table.insert(args, fb)
        -- axis mode: cos | principal | bbox
        local sam2Axes = { "cos", "principal", "bbox" }
        local axis = sam2Axes[sam2AxisIdx] or "cos"
        if axis ~= "cos" then
            table.insert(args, "--sam2-axis")
            table.insert(args, axis)
        end
        -- EMA over the raw logit map (damps the per-frame silhouette jitter).
        -- Only forward when the user actually typed a value (0.0..1.0);
        -- empty = CLI default (0.5).
        local ema = tonumber(tostring(sam2Ema or ""))
        if ema and ema >= 0.0 and ema <= 1.0 then
            table.insert(args, "--sam2-ema")
            table.insert(args, string.format("%.3f", ema))
        end
    end
    -- sot-mil / sot-csrt / sot-kcf: no external weights (mil builds its
    -- own classifier online; csrt/kcf are self-contained) - the two ONNX
    -- pickers are intentionally not forwarded for them.
    if useCalibrate then table.insert(args, "--calibrate") end
    if autoPartners then table.insert(args, "--auto-partners") end
    if noSceneRoi then table.insert(args, "--no-scene-roi") end
    if useVlm then table.insert(args, "--vlm") end
    -- ── LLM (Ollama) mode - forward the model + all knobs when tracker=ollama ─
    -- Same "forward only-when-relevant" pattern as yolo / dino / ostrack /
    -- gsam2 above: the CLI accepts these arguments unconditionally (they all
    -- have defaults), but the GUI keeps them visible and meaningful only
    -- when the user has picked 'ollama' as the tracker.
    if tracker == "ollama" then
        -- model (REQUIRED; placeholder / missing is still forwarded - the
        -- CLI will raise a clear error on the first run if empty)
        local om = (ollamaModels and ollamaModelIdx and ollamaModels[ollamaModelIdx]) or ""
        if om ~= "" and om:sub(1, 1) ~= "(" then
            table.insert(args, "--ollama-model")
            table.insert(args, om)
        end
        -- clip duration (seconds per sampled image); 0 = empty = CLI default
        local clip = tonumber(tostring(llmClipS or ""))
        if clip and clip > 0 then
            table.insert(args, "--llm-clip-s")
            table.insert(args, string.format("%.3f", clip))
        end
        -- stop time (MM:SS) - optional; empty string = end of video
        local stopT = tostring(llmStop or "")
        if stopT:match("%S") then
            table.insert(args, "--llm-stop")
            table.insert(args, stopT)
        end
        -- Ollama generate() options
        local nctx = tonumber(tostring(llmNumCtx or ""))
        if nctx and nctx > 0 then
            table.insert(args, "--num-ctx")
            table.insert(args, tostring(math.floor(nctx)))
        end
        local npred = tonumber(tostring(llmNumPredict or ""))
        if npred and npred > 0 then
            table.insert(args, "--num-predict")
            table.insert(args, tostring(math.floor(npred)))
        end
        local nbatch = tonumber(tostring(llmNumBatch or ""))
        if nbatch and nbatch > 0 then
            table.insert(args, "--num-batch")
            table.insert(args, tostring(math.floor(nbatch)))
        end
        local thumb = tonumber(tostring(llmThumb or ""))
        if thumb and thumb >= 32 then
            table.insert(args, "--llm-thumb")
            table.insert(args, tostring(math.floor(thumb)))
        end
        -- custom prompt (optional; empty = CLI default in core/config.py)
        local customP = tostring(llmPrompt or "")
        if customP:match("%S") then
            table.insert(args, "--llm-prompt")
            table.insert(args, customP)
        end
    end

    -- remember the context for the import (BEFORE launch: the "Kill" path
    -- and update() must never touch a genCtx that does not exist yet)
    genCtx = { scriptIdx = scriptIdx, t0 = startTime, t1 = endTime, outFile = outFile }
    lastError = ""

    -- Launch exactly like the working "Funscript Generator Windows" (MTFG):
    --    return Process.new(cmd, arg1, arg2, ...)
    -- direct launch, no shell. stdout/stderr go to OFS's pipes (visible in
    -- the OFS extension log window); the real traceback is always saved by
    -- the python side to the log file below, so "Open stderr log" has
    -- content even for a crash inside the imports.
    local logFile = extDir:gsub("/", "\\") .. "\\funscript_gen.stderr.log"
    if not fileExists(logFile) then
        local lf = io.open(logFile, "w")
        if lf then lf:close() end
    end

    -- quote arguments that contain spaces (video path, ext dir...)
    local parts = { pythonCmd }
    for _, a in ipairs(args) do
        local s = tostring(a)
        if s:find(" ") and s:sub(1, 1) ~= '"' then s = '"' .. s .. '"' end
        table.insert(parts, s)
    end
    print("[HF] launching: " .. table.concat(parts, " "))

    local ok, p, err = xpcall(function()
        return Process.new(pythonCmd, table.unpack(args))
    end, function(e)
        return string.format("%s\n%s", tostring(e), debug.traceback())
    end)
    if not ok or not p then
        procHandle = nil
        lastError = "ERROR: cannot start python: " .. tostring(err)
        status = lastError
        print(lastError)
        return
    end
    procHandle = p
    status = "Generating... (" .. tracker .. ")"
end

-- ── robust: normalize whatever join() returns to a plain numeric "code" ──

local function normalizeCode(r)
    -- possible return shapes across sol bindings:
    --   number (exit code)
    --   { code = <num>, status = <str> }  (detached process)
    --   { exitCode = <num> }  (other wrapper names)
    --   table with numeric [1]
    if type(r) == "number" then return r end
    if type(r) == "table" then
        if r.code and type(r.code) == "number" then return r.code end
        if r.exitCode and type(r.exitCode) == "number" then return r.exitCode end
        if r[1] and type(r[1]) == "number" then return r[1] end
    end
    if type(r) == "string" and r:match("^%d+$") then return (r:tonumber() or r) end
    return nil -- nil means "unknown / process did not actually exit cleanly"
end

-- ── open the python stderr log in Notepad (for quick debugging) ──

-- single log file: python writes it directly (funscript_gen.py _file_log);
-- the GUI "Open stderr log" button and the tail preview read the same file
local LOG_NAME = "funscript_gen.stderr.log"

function openLog()
    local log = ofs.ExtensionDir() .. "\\" .. LOG_NAME
    print("[HF] opening log: " .. log)
    local ok, err = pcall(Process.new, "notepad.exe", log)
    if not ok then
        lastError = "could not open the log file: " .. tostring(err)
    end
    return ok
end

-- ── return the last N lines of the log (for the in-GUI preview) ──

function readLogTail(n)
    n = n or 15
    local out = {}
    local f = io.open(ofs.ExtensionDir() .. "\\" .. LOG_NAME, "r")
    if f then
        local content = f:read("*a") or ""
        f:close()
        local lines = {}
        for line in content:gmatch("[^\r\n]*") do
            if line ~= "" then table.insert(lines, line) end
        end
        local from = math.max(1, #lines - n + 1)
        for i = from, #lines do out[#out + 1] = lines[i] end
    end
    return out
end

-- ── import the result ──

function importResult()
    if not genCtx then return end

    local f = io.open(genCtx.outFile, "r")
    if not f then
        status = "ERROR: result file not found"
        return
    end
    local content = f:read("*a")
    f:close()

    local body = json.decode(content)
    if not body or not body.actions then
        status = "ERROR: bad JSON in result file"
        return
    end

    local script = ofs.Script(genCtx.scriptIdx)

    -- OFS invariant: ONE action per millisecond-timestamp (Serialize keys on
    -- round(atS*1000)).  Dedup therefore uses INTEGER milliseconds as the key
    -- (NOT 1/fps, which over-merges: at 30fps a whole frame = 33ms, so the old
    -- "closest within 1 frame" filter silently dropped ~half the points).
    -- Step 1: self-dedupe the generated batch (the batch is already sorted by
    -- time from Python, so the first point of a given timestamp wins).
    local batch, seen = {}, {}
    local selfDropped = 0
    for _, a in ipairs(body.actions) do
        local ms = math.floor(tonumber(a.at) + 0.5)
        if not seen[ms] then
            seen[ms] = true
            table.insert(batch, { ms = ms, pos = a.pos })
        else
            selfDropped = selfDropped + 1
        end
    end

    -- Step 2: key the points already on the timeline (integer ms) so we do not
    -- create a duplicate-timestamp pair -- that is exactly what makes Commit()
    -- abort ("multiple actions with the same timestamp") and drop the WHOLE
    -- batch (one duplicate used to kill the entire import).
    local existing = {}
    for _, a in ipairs(script.actions) do
        existing[math.floor(a.at * 1000.0 + 0.5)] = true
    end

    -- Step 3: insert every surviving point exactly once (kept selected).
    local added, crossDropped = 0, 0
    for _, a in ipairs(batch) do
        if not existing[a.ms] then
            existing[a.ms] = true
            local pos = a.pos
            if pos < 0 then pos = 0 end
            if pos > 100 then pos = 100 end
            script.actions:add(Action.new(a.ms / 1000.0, pos, true)) -- true = selected
            added = added + 1
        else
            crossDropped = crossDropped + 1
        end
    end

    -- Step 4: commit, guarded -- a residual duplicate (float-round edge) no
    -- longer aborts the whole import silently.
    script:sort()
    local ok, err = pcall(function() script:commit() end)
    if not ok then
        addedCount = 0
        status = "ERROR on commit: " .. tostring(err)
        lastError = "(see the \"Open stderr log\" button)"
        print("[HF] commit failed: " .. tostring(err))
        return
    end

    addedCount = added
    -- canonical PASSED status: the OFS GUI (main.lua) uses the "PASSED:"
    -- prefix in `status` to show the single-line success row next to the
    -- "Open stderr log" button (mirrors funscript_gen.py's "PASSED:" log
    -- line - the two keep the same marker so debugging stays consistent).
    status = "PASSED: " .. added .. " actions added"
    local dropped = selfDropped + crossDropped
    if dropped > 0 then
        status = status .. "  (" .. dropped .. " duplicate-ms merged: "
            .. selfDropped .. " self, " .. crossDropped .. " existing)"
    end
    -- When the RDP window was closed by the user (X) - vs the user pressing
    -- Enter for the OK path - add a note so they know that no range trim or
    -- RDP was applied on this run (the log has the full story).
    local lf = io.open(ofs.ExtensionDir() .. "\\" .. LOG_NAME, "r")
    if lf then
        local lc = lf:read("*a") or ""
        lf:close()
        if lc:find("RDP window was closed by the user") then
            status = status .. "  (RDP window was closed by you - full unscaled track, no RDP applied)"
        end
    end
    print(("[HF] imported %d actions (%d self-dupes, %d cross-dupes)"):format(
        added, selfDropped, crossDropped))
end

-- ── mandatory extension functions ──

function init()
    print("[HF] extension loaded, dir: " .. ofs.ExtensionDir())
    status = "Idle"
    loadSettings()
end

function update(delta)
    if procHandle then
        if procHandle:alive() then
            status = "Generating... (q/space in the preview window = stop and save)"
        else
            -- the process has finished
            local ok, res = pcall(function() return procHandle:join() end)
            procHandle = nil
            local code = ok and normalizeCode(res) or nil
            if code == 0 then
                local ok2, r = pcall(importResult)
                if not ok2 then
                    status = "ERROR: import crashed"
                    lastError = tostring(r)
                    print("[HF] importResult error: " .. tostring(r))
                end
            elseif code == 2 then
                -- user aborted (identify / calibrate window)
                status = "Aborted by the user - nothing imported"
                lastError = ""
            else
                -- a real error, import the partial result if any
                local f = io.open(genCtx and genCtx.outFile or "", "r")
                local partial = f and (f:read(1) ~= nil)
                if f then f:close() end
                local importErr = nil
                if partial then
                    local ok2, r = pcall(importResult)
                    if not ok2 then importErr = tostring(r) end
                end
                if partial and addedCount > 0 then
                    status = "ERROR (imported partial result: "
                        .. addedCount .. " actions), code " .. tostring(code)
                else
                    status = "ERROR: python exited with code " .. tostring(code)
                end
                if importErr then
                    status = status .. "  [import error: " .. importErr .. "]"
                end
                lastError = "(see the \"Open stderr log\" button - traceback is saved there)"
            end
        end
    end
end

function gui()
    local anyChanged = false
    ofs.Text("Hismith Funscript Generator")
    ofs.Separator()

    ofs.Text("Status: " .. status)
    if lastError ~= "" then
        ofs.Text(lastError)
    end
    if addedCount > 0 then
        ofs.Text("Last run: " .. addedCount .. " actions (selected)")
    end
    if status:find("ERROR") or status:find("PASSED") then
        if ofs.Button("Open stderr log##openLogBtn") then
            openLog()
        end
        ofs.Tooltip(
            "Opens funscript_gen.stderr.log in Notepad - the full log (traceback on error, full pipeline log on success). The line below is ONLY the LAST canonical status line (ERROR: ... or PASSED: ...).")
        -- show ONLY the final status line (the canonical "ERROR: ..." /
        -- "PASSED: ..." line that funscript_gen.py writes as the LAST
        -- line of the log) - not the whole 15-line tail.  The full
        -- log is one click away in Notepad.
        local last = readLogTail(1)
        if #last > 0 then
            ofs.Text(last[1])
        else
            ofs.Text("(log file is empty - python did not even start / crash before import)")
        end
    end
    ofs.Separator()

    -- parameters
    prevTrackerIdx = trackerIdx
    trackerIdx, c = ofs.Combo("Tracker", trackerIdx,
        { "yolo - pose (default)",
            "mediapipe - pose (33 landmarks)",
            "sot-csrt - OpenCV single-object tracker (ROI)",
            "sot-kcf - OpenCV single-object tracker (ROI)",
            "sot-vit - OpenCV ViT tracker (ROI)",
            "sot-mil - OpenCV MIL tracker (ROI)",
            "groundingdino - text-prompt boxes",
            "ostrack - ViT one-stream, max accuracy (ONNX)",
            "gsam2 - Grounded SAM 2, pixel-mask silhouette (max accuracy)",
            "ollama - LLM: draw a ROI, the model reads all sampled frames (slow, for hard clips)" })
    if c then anyChanged = true end
    -- (tooltip below)
    ofs.Tooltip(
        "yolo / mediapipe: multi-person pose - the Identify window picks active/passive partners. SOT (csrt / kcf / vit / mil): single object - draw an ROI box around the exact body zone (Enter to start, R in the preview to re-select when lost); robust, but only ONE person can be tracked. groundingdino: text-prompt open-vocabulary detection. ostrack: OSTrack - the most accurate single-object tracker, ONNX weights. sot-vit: OpenCV TrackerVit. gsam2: Grounded SAM 2 (pixel-mask silhouette). ollama: LLM (VLM) reads sampled frames of a single ROI region and outputs a depth-per-frame JSON track — slow, but best for 'hard' segments; requires a running `ollama serve` with a MULTIMODAL model, plus one extra interactive window on start to draw the ROI.")
    if trackerIdx == 1 then
        yoloModelIdx, c = ofs.Combo("YOLO model", yoloModelIdx,
            { "yolov8n-pose", "yolov8s-pose", "yolov8m-pose", "yolov8l-pose", "yolov8x-pose",
                "yolo11n-pose", "yolo11s-pose", "yolo11m-pose", "yolo11l-pose", "yolo11x-pose",
                "yolo26n-pose", "yolo26s-pose", "yolo26m-pose", "yolo26l-pose", "yolo26x-pose" })
        if c then anyChanged = true end
        ofs.Tooltip(
            "default: yolo26x-pose (latest family: end-to-end NMS-free detection). n=fast/rough ... x=max. The model downloads automatically on first run. YOLO input size is 1280 (accurate on 1080p).")
    end
    if trackerIdx == 7 then
        dinoModelIdx, c = ofs.Combo("DINO model", dinoModelIdx,
            { "Swin-B (flagship, 56.7 AP) - default",
                "Swin-T (lighter, 48.4 AP)" })
        if c then anyChanged = true end
        ofs.Tooltip(
            "GroundingDINO checkpoint. Swin-B (default) is the most accurate build (trained additionally on RefCOCO / OpenImages); Swin-T is lighter and faster and works the same way. Weights download automatically on first run (Swin-B is ~895 MB).")
        -- DINO target list is queried from Python (--list dino = the
        -- DINO_TARGETS in core/detectors/groundingdino.py, the single
        -- source of truth) and CACHED (ensureDinoTargets): no python
        -- process per draw frame.
        if prevTrackerIdx ~= 7 then
            ensureDinoTargets(ofs.ExtensionDir(), pythonCmd)
        end
        if (dinoTargetIdx or 1) > #dinoTargets then
            dinoTargetIdx = 1 -- selection no longer valid
        end
        dinoTargetIdx, c = ofs.Combo("GroundingDINO target", dinoTargetIdx, dinoTargets)
        if c then anyChanged = true end
        ofs.Tooltip(
            "The detection target (a named text prompt) - INDEPENDENT of the pose-tracker 'point' list. Chosen here from the list provided by core/detectors/groundingdino.py (the single source of truth). Point p (where inside the detected box) is then chosen interactively in the ROI window - default = box center; press+hold LMB inside the active box to move it, release to fix. The prompt is the same color as the person boxes in that window.")
        dinoCustomPrompt, c = ofs.Input("Custom GroundingDINO prompt", dinoCustomPrompt)
        if c then anyChanged = true end
        ofs.Tooltip(
            "OPTIONAL: when non-empty, REPLACES the selected target's built-in phrase with an exact custom text prompt (e.g. 'the waistband line of a pair of shorts'). Leave empty to use the named target's prompt. Applies to groundingdino only.")
    end
    -- ── OSTrack ONNX picker - visible ONLY for the 'ostrack' tracker ─
    -- (mirrors the YOLO/DINO combos: the model list is shown only when the
    -- selected tracker consumes that folder).  List is CACHED
    -- (ensureFamilyModels "ostrack"): no `dir /b` on idle frames.  Re-scan
    -- fires only when a cached file disappears, or unconditionally at
    -- launch (see startGeneration).
    if trackerIdx == 8 then
        if prevTrackerIdx ~= 8 then
            ensureFamilyModels(ofs.ExtensionDir(), "ostrack")
        end
        if (ostrackModelIdx or 1) > #ostrackModels + 1 then
            ostrackModelIdx = 1 -- selection no longer valid
        end
        local ostrackOpts
        if #ostrackModels == 0 then
            -- empty folder: the placeholder REPLACES the "auto" entry (a
            -- non-functional "auto" would be misleading)
            ostrackOpts = { "(models/ostrack/ folder is empty)" }
            ostrackModelIdx = 1
        else
            ostrackOpts = { "auto (first in models/ostrack/)" }
            for _, n in ipairs(ostrackModels) do table.insert(ostrackOpts, n) end
        end
        ostrackModelIdx, c = ofs.Combo("OSTrack model (onnx)", ostrackModelIdx, ostrackOpts)
        if c then anyChanged = true end
        ofs.Tooltip(
            "OSTrack - ONNX weights for the InferenceSession " ..
            "(template + search inputs, score/size/offset outputs).\n" ..
            "  * 'auto' = the first .onnx in models/ostrack/.\n" ..
            "  * any file listed = used as-is (cv2 TracerVit-style params are " ..
            "not involved).\n" ..
            "  * 'folder is empty' = put a *.onnx file into models/ostrack/ " ..
            "(no official ONNX release: botaoye/OSTrack exports + the " ..
            "l-sf/Track-trt recipe).")
    end
    -- ── sot-vit ONNX picker - visible ONLY for the 'sot-vit' tracker ─
    if trackerIdx == 5 then
        if prevTrackerIdx ~= 5 then
            ensureFamilyModels(ofs.ExtensionDir(), "vittrack")
        end
        if (vitModelIdx or 1) > #vitModels + 1 then
            vitModelIdx = 1
        end
        local vitOpts
        if #vitModels == 0 then
            -- empty folder: the placeholder REPLACES the "auto" entry
            -- (auto-download of vitTracker.onnx still happens on first run)
            vitOpts = { "(models/vittrack/ folder is empty)" }
            vitModelIdx = 1
        else
            vitOpts = { "auto (first in models/vittrack/)" }
            for _, n in ipairs(vitModels) do table.insert(vitOpts, n) end
        end
        vitModelIdx, c = ofs.Combo("sot-vit model (onnx)", vitModelIdx, vitOpts)
        if c then anyChanged = true end
        ofs.Tooltip(
            "sot-vit (OpenCV TrackerVit) - TrackerVit ONNX weights " ..
            "(applied via cv2.TrackerVit_Params().net).\n" ..
            "  * 'auto' = auto-download of vitTracker.onnx into " ..
            "models/vittrack/ then the CWD (legacy behaviour).\n" ..
            "  * any file listed = used as-is.\n" ..
            "  * 'folder is empty' = the auto-download will fetch " ..
            "vitTracker.onnx on the first run, or drop a .onnx file into " ..
            "models/vittrack/.")
    end
    local isSotTracker = (trackerIdx >= 3 and trackerIdx <= 6)
        or trackerIdx == 8 -- ostrack
        or trackerIdx == 9 -- gsam2 (Grounded SAM2)
    local isDinoTracker = (trackerIdx == 7)
    local isLlmTracker = (trackerIdx == 10) -- ollama - LLM mode (no "point" combo)
    -- ── Grounded SAM 2 (gsam2) checkpoint + point-p axis - ONLY for idx 9 ──
    if trackerIdx == 9 then
        -- checkpoint list from Python (--list sam2, the single source of
        -- truth = core/detectors/models.py SAM2_VARIANTS, default FIRST).
        -- LAZY + CACHED (ensureSam2Models): no python process on idle
        -- draw frames (same pattern as ensureDinoTargets / ensureFamilyModels).
        if prevTrackerIdx ~= 9 then
            ensureSam2Models(ofs.ExtensionDir(), pythonCmd)
        end
        if (sam2ModelIdx or 1) > #sam2Models then
            sam2ModelIdx = 1 -- selection no longer valid
        end
        sam2ModelIdx, c = ofs.Combo("SAM2 model", sam2ModelIdx, sam2Models)
        if c then anyChanged = true end
        ofs.Tooltip("Grounded SAM 2 checkpoint (facebookresearch/sam2). " ..
            "sam2.1-hiera-base-plus is the default (best accuracy/size balance). " ..
            "sam2.1-hiera-large = max accuracy (~8 GB VRAM, fits the RTX 5090). " ..
            "sam2.1-hiera-tiny = fastest / smallest. Weights auto-download into " ..
            "models/sam2/ on first run (see core/detectors/models.py SAM2_VARIANTS).")
        sam2AxisIdx, c = ofs.Combo("SAM2 point axis (p)", sam2AxisIdx,
            { "center of mass (cos) - default, rotation-invariant",
                "center of mass (principal) - same signal, debug knob",
                "bounding-box center (bbox) - box-tracker style" })
        if c then anyChanged = true end
        ofs.Tooltip("How the tracked point p is derived from the silhouette mask. " ..
            "cos / principal = subpixel CENTER OF MASS - the rotation-invariant " ..
            "point (recommended; a changing camera angle does not shift it). " ..
            "bbox = the mask's bounding-box center (same rule as the other SOT " ..
            "trackers; a debug knob).")
        sam2FramesIdx, c = ofs.Combo("SAM2 frames backend", sam2FramesIdx,
            { "lazy - per-frame decord decode (constant RAM, RECOMMENDED)",
                "jpeg - pre-export + async loader (RAM grows with clip)" })
        if c then anyChanged = true end
        ofs.Tooltip("How SAM 2 receives video frames. " ..
            "'lazy' (default) decodes each frame ON DEMAND via decord, so RAM " ..
            "stays constant (a few frames) regardless of clip length - this " ..
            "removes the upfront whole-clip RAM spike (a ~2 min 1080p clip " ..
            "is ~47 GB, a 5-min video ~95 GB) that sits between start and the " ..
            "first mask. 'jpeg' pre-exports the whole video to a JPEG folder " ..
            "and lets SAM 2's built-in async loader stream from it - faster " ..
            "decode for some codecs, but RAM still grows with clip length " ..
            "(useful as a comparison baseline).")
        sam2Ema, c = ofs.Input("SAM2 EMA (0.0-1.0)", sam2Ema)
        if c then anyChanged = true end
        ofs.Tooltip("EMA blend factor over the raw SAM 2 logit map, applied " ..
            "BEFORE the >0 binarisation. Dampens the per-frame " ..
            "silhouette-boundary 'breathing' that shows up as a 10-30 px " ..
            "centroid oscillation around the ROI center (visible in " ..
            "funscript_gen.stderr.log as score:1.0 on every frame). " ..
            "1.000 = OFF (legacy, sharpest tracking). 0.500 (default) " ..
            "halves the weight of any single noisy frame. " ..
            "0.700-0.800 = smoother but slightly lagged. " ..
            "0.200 = sharper tracking than default. " ..
            "The reference peak used by the loss score is calibrated on " ..
                "the FIRST accepted frame after init() - it is independent of " ..
                "this knob.")
        end
        -- ── LLM (Ollama) generation mode - ONLY for idx 10 ──
        -- One single fixed ROI on ONE frame of the video.  Every sampled frame
        -- of the selected span is cropped by that ROI, thumbnail'd and base64
        -- encoded, then sent in ONE multimodal `generate()` call to the
        -- selected Ollama model.  The model answers a STRICT JSON array of
        -- {frame, depth_pct} objects; the (t_sec, pos) track is built
        -- client-side (time = start_s + frame_index * clip_s, pos = depth_pct
        -- directly: 0%=inside / 100%=outside in the PROJECT's terms).
        -- All the defaults mirror core/config.py LLM_DEFAULT_*:
        --   clip_s = 0.5, num_ctx = 200000, num_predict = 32000,
        --   num_batch = 2048, thumb = 1920.
        if isLlmTracker then
            -- model list: LAZY + CACHED (ensureOllamaModels), same pattern as
            -- ensureDinoTargets.  The list comes from `ollama list` (a running
            -- `ollama serve` is required; the Python client raises a clean
            -- error and the CLI will show the user-friendly message).
            if prevTrackerIdx ~= 10 then
                ensureOllamaModels(ofs.ExtensionDir(), pythonCmd)
            end
            if (ollamaModelIdx or 1) > #ollamaModels then
                ollamaModelIdx = 1 -- selection no longer valid
            end
            ollamaModelIdx, c = ofs.Combo("Ollama model", ollamaModelIdx, ollamaModels)
            if c then anyChanged = true end
            ofs.Tooltip(
                "The Ollama model to run for the LLM (VLM) generation mode. " ..
                "This MUST be a MULTIMODAL (vision-capable) model that can see " ..
                "images - a plain text LLM will not follow the JSON-in-the-prompt " ..
                "instruction.  List comes from `ollama list`; a running `ollama " ..
                "serve` is required.  The model name (with any tag, e.g. " ..
                "my-model:32b) is forwarded to the CLI as `--ollama-model`.  " ..
                "Defaults are 128k context / 32k response / 2k batch / 1920px " ..
                "thumbnail / 0.5s clip spacing - the knobs below expose each.")
            -- clip duration (seconds per sampled image).  Lower = finer
            -- temporal resolution; higher = longer spans fit in one request.
            llmClipS, c = ofs.Input("LLM clip duration (s per image)", llmClipS)
            if c then anyChanged = true end
            ofs.Tooltip(
                "How many seconds of video = ONE sampled image sent to the LLM. " ..
                "Default 0.5 (one image every 0.5 s → a 30 s clip = 60 images). " ..
                "Bigger clip = fewer images (faster, less VRAM); smaller clip = " ..
                "finer temporal resolution (slower, more detail).  The MAXIMUM " ..
                "is the MAX_IMAGES_PER_CALL=64 cap in core/postpass/llm_gen.py - " ..
                "any span that exceeds 64 samples with the current clip_s is " ..
                "TRUNCATED to the first 64 (a warning is logged).  For a 2 min " ..
                "clip use ~2.0 s per image to stay under the cap.")
            -- stop time (optional; empty = end of video)
            llmStop, c = ofs.Input("LLM stop time (MM:SS, empty = end of video)", llmStop)
            if c then anyChanged = true end
            ofs.Tooltip(
                "Optional: end of the LLM analysis span (MM:SS in ABSOLUTE " ..
                "video time, the same format as the start time).  When empty, " ..
                "the span runs from the [start] value to the end of the video. " ..
                "Only the frames in [start .. stop] are sampled and sent to the " ..
                "LLM - use this to limit generation to a 'hard' segment that " ..
                "the tracker missed.")
            -- Ollama `generate()` options (the three knobs the reference
            -- implementation exposed; see plans/llm-ollama-generation.md)
            llmNumCtx, c = ofs.Input("Ollama num_ctx (context budget)", llmNumCtx)
            if c then anyChanged = true end
            ofs.Tooltip(
                "Ollama explicit context budget in TOKENS (default 200000). " ..
                "A 1920x1920 JPEG image occupies ~1500-2500 tokens depending on " ..
                "the model, so 128k ctx holds ~50-80 images + the prompt + the " ..
                "response.  Bigger costs VRAM; smaller is faster.  On an RTX " ..
                "5090 (32 GB) any value up to 256k usually fits.")
            llmNumPredict, c = ofs.Input("Ollama num_predict (max response)", llmNumPredict)
            if c then anyChanged = true end
            ofs.Tooltip(
                "Ollama max response length in TOKENS (default 32000).  The " ..
                "model may emit a 'thinking' passage before the JSON array; " ..
                "this caps the total.  Must be larger than the JSON array " ..
                "itself (at ~50 chars per entry * 64 images ≈ 4k tokens, so " ..
                "32k is a comfortable headroom).")
            llmNumBatch, c = ofs.Input("Ollama num_batch (prompt batch)", llmNumBatch)
            if c then anyChanged = true end
            ofs.Tooltip(
                "Ollama prompt-processing batch size (default 2048).  Larger " ..
                "values speed up the initial image-encoding pass on a GPU " ..
                "(a 64-image batch at 2048 is ~30 s on RTX 5090; at 512 it " ..
                "is ~100 s).  Smaller values cap VRAM.")
            llmThumb, c = ofs.Input("Thumbnail max side (px)", llmThumb)
            if c then anyChanged = true end
            ofs.Tooltip(
                "Maximum side in pixels of each cropped image BEFORE encoding " ..
                "(1920 default, aspect preserved, downscale-only).  Smaller = " ..
                "smaller base64 payload + faster encoding + fits more in " ..
                "context; too small (e.g. <200) loses the detail the VLM needs " ..
                "to judge depth.  Use 1920-640 for most clips; the 1920 default " ..
                "is the same as the reference implementation.")
            -- optional prompt override (default is in core.config.LLM_DEFAULT_PROMPT)
            llmPrompt, c = ofs.Input("Custom LLM prompt (empty = default)", llmPrompt)
            if c then anyChanged = true end
            ofs.Tooltip(
                "OPTIONAL: when non-empty, REPLACES the default depth-of-" ..
                "penetration prompt with this exact string.  The prompt MUST " ..
                "demand the strict JSON shape [{\"frame\": N, \"depth_pct\": X}, " ..
                "...] for the parser to work.  Leave empty to use the built-in " ..
                "default (see core/config.py LLM_DEFAULT_PROMPT).")
        end
        if not isSotTracker and not isDinoTracker and not isLlmTracker then
        pointIdx, c = ofs.Combo("Tracked point (p)", pointIdx,
            { "mid-hips (between both hips) - default",
                "left-hip (the person's own left hip)",
                "right-hip (the person's own right hip)",
                "thigh (midpoint of both knees)",
                "torso (center between hips and shoulders)",
                "shoulders (midpoint of both shoulders)",
                "head (top of the head)",
                "ankle (midpoint of both ankles)" })
        if c then anyChanged = true end
        ofs.Tooltip(
            "Which body zone is tracked as the point p. mid-hips / left-hip / right-hip cover the pelvis zone - pick the single side that gives the most stable movement. Recalibrate f/b after changing it. Hold S in any window to see the skeleton and exactly where p is. (Hidden for SOT trackers - they track the CENTER of the ROI you draw. Hidden for GroundingDINO - its target list is separate ('GroundingDINO target' above) and point p is picked interactively in the ROI window.)")
    end
    flowIdx, c = ofs.Combo("Point smoothing (anti-jitter)", flowIdx,
        { "off - pose point as-is (default)",
            "pure - optical flow, pose rebind only on loss",
            "fusion - pose anchor + flow particles" })
    if c then anyChanged = true end
    ofs.Tooltip(
        "Optical-flow refinement of the tracked point p (reduces jitter when the pose model jumps). In the preview: green dot = final point, red ring = raw pose anchor (only when they differ). 'fusion' is recommended if the point skips around; 'pure' may drift on textureless areas.")
    useCalibrate, c = ofs.Checkbox("Calibrate f/b manually (interactive window)", useCalibrate)
    if c then anyChanged = true end
    ofs.Tooltip(
        "On: interactive window - mark f (full-in, 0%) and b (full-out, 100%) on a frame of the active partner; these points define the thrust axis. Off: the f/b axis is derived automatically from the spread of point p over the first ~6 seconds (rougher).")
    autoPartners, c = ofs.Checkbox("Auto partners (skip the selection window)", autoPartners)
    if c then anyChanged = true end
    ofs.Tooltip(
        "If enabled - the most confident detected person becomes the active partner, the second one the passive (no Identify window).")
    if isSotTracker then
        noSceneRoi, c = ofs.Checkbox("SOT: skip the scene ROI setup (full frame as-is)", noSceneRoi)
        if c then anyChanged = true end
        ofs.Tooltip(
            "ON - the scene setup window (zoom 1x/2x/3x, rotate 15deg steps, pan 10px, optional scene-ROI crop) is SKIPPED: the video runs on the full frame with identity transform (fastest path, no extra transformation). OFF (default) - the scene setup window opens BEFORE the ROI-drag window; whatever transform you pick is applied to every frame the tracker sees.")
    end
    useVlm, c = ofs.Checkbox("VLM scene analysis (Ollama)", useVlm)
    if c then anyChanged = true end
    minIntervalMs, c = ofs.InputInt("Min interval (ms)", minIntervalMs, 10)
    if c then anyChanged = true end
    latencyMs, c = ofs.InputInt("Latency (ms)", latencyMs, 10)
    if c then anyChanged = true end

    ofs.Separator()
    ofs.Text("How to interpret the motion")
    motionIdx, c = ofs.Combo("Motion mode", motionIdx,
        { "straight (linear projection onto the f→b axis)",
            "arc (fitted arc, progress along the arc)",
            "parabola (fitted parabola, progress along it)",
            "complex (real trajectory of point p, raw path)" })
    if c then anyChanged = true end
    ofs.Tooltip(
        "straight - projection onto the axis; arc/parabola - curve fit, progress measured by arc length; complex - the actual per-frame path length of point p (recommended for real video).")
    normalizeIdx, c = ofs.Combo("Normalize", normalizeIdx,
        { "per-stroke: each stroke = 100%",
            "global-max: normalize against the largest stroke of the session" })
    if c then anyChanged = true end
    ofs.Tooltip(
        "per-stroke: every f→b / b→f stroke goes 0→100 independently. global-max: the longest stroke of the whole generation = 100%, the others are scaled proportionally.")

    ofs.Separator()
    pythonCmd, c = ofs.Input("Python cmd", pythonCmd)
    if c then anyChanged = true end

    ofs.BeginDisabled(procHandle ~= nil)
    if ofs.Button("Generate from here##genBtn") then
        startGeneration()
    end
    ofs.EndDisabled()
    ofs.Tooltip(
        "Generate from the current player position to the end of the video. Stop: q/space in the preview window (saves the processed part).")

    if procHandle then
        ofs.Separator()
        if ofs.Button("Kill process (emergency)##killBtn") then
            local h = procHandle
            procHandle = nil
            h:kill()
            -- try to import whatever has been written so far
            local f = io.open(genCtx and genCtx.outFile or "", "r")
            local partial = f and (f:read(1) ~= nil)
            if f then f:close() end
            if partial then
                local ok2, r = pcall(importResult)
                if not ok2 then
                    status = "Killed - import error: " .. tostring(r)
                    lastError = tostring(r)
                elseif addedCount > 0 then
                    status = "Killed - partial result imported ("
                        .. addedCount .. " actions)"
                else
                    status = "Killed"
                end
            else
                status = "Killed - no partial result available"
            end
        end
    end
    if anyChanged then
        saveSettings()
    end
end
