from matplotlib.colors import to_rgb


class Color:

    def __init__(self, red: int, green: int, blue: int):
        self.red = Color.clamp_intensity(red)
        self.green = Color.clamp_intensity(green)
        self.blue = Color.clamp_intensity(blue)

    @staticmethod
    def clamp_intensity(value: int) -> int:
        return min(max(value, 0), 255)

    @classmethod
    def from_string(cls, color_name: str, alpha: float = 1.0):
        red, green, blue = map(lambda value: int(round(255 * ((value - 1) * alpha + 1))), to_rgb(color_name))
        return cls(red=red, green=green, blue=blue)

    def hex_code(self) -> str:
        return f"#{''.join('{:02x}'.format(value) for value in (self.red, self.green, self.blue))}"


