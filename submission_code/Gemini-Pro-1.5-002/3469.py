class Solution:
    def maxHeightOfTriangle(self, red: int, blue: int) -> int:
        dp = {}

        def solve(r, b, level, color):
            if (r, b, level, color) in dp:
                return dp[(r, b, level, color)]

            balls_needed = level
            if color == 'r':
                if r < balls_needed:
                    return 0
                
                ans = level
                
                rem_r = r - balls_needed
                ans = max(ans, solve(rem_r, b, level + 1, 'b'))
                
            else:
                if b < balls_needed:
                    return 0
                
                ans = level
                rem_b = b - balls_needed
                ans = max(ans, solve(r, rem_b, level + 1, 'r'))
                
            dp[(r, b, level, color)] = ans
            return ans

        return max(solve(red, blue, 1, 'r'), solve(red, blue, 1, 'b'))