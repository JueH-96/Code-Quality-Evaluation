class Solution:
    def theMaximumAchievableX(self, num: int, t: int) -> int:
        # Each operation can change the difference between x and num by ±2.
        # To make x equal to num, the difference (x - num) must be neutralized.
        # To maximize x = num + d, we want as large a d as possible, but d must be even,
        # and we can decrease d by 2 in one operation.
        # Hence, at most t operations can decrease the difference by 2*t.
        # So the maximum initial difference allowed is 2*t, and thus the largest achievable x is:
        #      x = num + 2*t.
        return num + 2 * t

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Test Example 1:
    print(sol.theMaximumAchievableX(4, 1))  # Expected output: 6
    # Test Example 2:
    print(sol.theMaximumAchievableX(3, 2))  # Expected output: 7