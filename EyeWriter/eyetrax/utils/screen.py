from screeninfo import get_monitors


def get_screen_size(screen_index: int = 0):
    m = get_monitors()[screen_index]
    return m.width, m.height
