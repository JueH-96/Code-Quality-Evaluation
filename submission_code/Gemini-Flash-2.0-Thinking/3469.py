class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        def calculate_needed(h, start_red):
            red_needed = 0
            blue_needed = 0
            for i in range(1, h + 1):
                if start_red:
                    if i % 2 == 1:
                        red_needed += i
                    else:
                        blue_needed += i
                else:
                    if i % 2 == 1:
                        blue_needed += i
                    else:
                        red_needed += i
            return red_needed, blue_needed

        max_possible_balls = red + blue
        max_h = 0
        h = 0
        while True:
            h += 1
            if h * (h + 1) // 2 > max_possible_balls:
                max_h = h - 1
                break
            if h > 200:  # Safety break
                max_h = h - 1
                break

        for h in range(max_h, 0, -1):
            # Try starting with red
            red_needed, blue_needed = calculate_needed(h, True)
            if red_needed <= red and blue_needed <= blue:
                return h

            # Try starting with blue
            red_needed, blue_needed = calculate_needed(h, False)
            if red_needed <= red and blue_needed <= blue:
                return h

        return 0