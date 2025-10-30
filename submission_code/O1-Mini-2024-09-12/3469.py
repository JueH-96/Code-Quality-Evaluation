class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        def build_triangle(start_color, red, blue):
            height = 0
            current_color = start_color
            required_balls = 1
            r, b = red, blue
            while True:
                if current_color == 'red':
                    if r >= required_balls:
                        r -= required_balls
                    else:
                        break
                else:
                    if b >= required_balls:
                        b -= required_balls
                    else:
                        break
                height += 1
                required_balls += 1
                current_color = 'blue' if current_color == 'red' else 'red'
            return height

        height_start_red = build_triangle('red', red, blue)
        height_start_blue = build_triangle('blue', red, blue)
        return max(height_start_red, height_start_blue)