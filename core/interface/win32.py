"""
Win32 key constants (OpenCV waitKeyEx codes carry the key state in the
upper byte) and the key-held helper.
"""

# OpenCV waitKeyEx: key code | (state << 16)
KEY_CTRL = 0x11        # win32 VK_CONTROL
KEY_LEFT = 0x25 << 16
KEY_RIGHT = 0x27 << 16
KEY_UP = 0x26 << 16
KEY_DOWN = 0x28 << 16


def key_held(vk: int) -> bool:
    """True while the given win32 VK code is physically pressed.

    MS docs: bit 15 (0x8000) = current state (pressed / unpressed);
    bit 0 (0x0001) = "state changed since the last PeekMessage/GetMessage" —
    which cv2.waitKeyEx flushes via the message queue, so using 0x0001
    makes the skeleton flash only on the press frame and then vanish.
    """
    try:
        from ctypes import windll
        return bool(windll.user32.GetAsyncKeyState(vk) & 0x8000)
    except Exception:
        return False
