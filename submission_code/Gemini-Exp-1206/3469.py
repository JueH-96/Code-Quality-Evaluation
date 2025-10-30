class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        def solve(r, b, prev_color, height):
            if r < 0 or b < 0:
                return 0
            
            if height == 0:
                return max(solve(r - 1, b, 0, 1), solve(r, b - 1, 1, 1))
            
            if prev_color == 0:
                return 1 + solve(r, b - (height + 1), 1, height + 1)
            else:
                return 1 + solve(r - (height + 1), b, 0, height + 1)

        def solve_iterative(red, blue):
            ans = 0
            for start_color in [0, 1]:
                r, b = red, blue
                height = 0
                prev_color = start_color
                
                while True:
                    if height == 0:
                        if prev_color == 0:
                            if r >= 1:
                                r -= 1
                                height = 1
                                prev_color = 1
                            else:
                                break
                        else:
                            if b >= 1:
                                b -= 1
                                height = 1
                                prev_color = 0
                            else:
                                break
                    else:
                        if prev_color == 0:
                            if b >= height + 1:
                                b -= (height + 1)
                                height += 1
                                prev_color = 1
                            else:
                                break
                        else:
                            if r >= height + 1:
                                r -= (height + 1)
                                height += 1
                                prev_color = 0
                            else:
                                break
                ans = max(ans, height)
            return ans

        return solve_iterative(red, blue)