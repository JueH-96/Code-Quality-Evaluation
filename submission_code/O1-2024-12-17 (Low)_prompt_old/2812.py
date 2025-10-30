class Solution:
    def theMaximumAchievableX(self, num: int, t: int) -> int:
        # Explanation:
        # Each operation can shift x and num by 1 in opposite directions.
        # To get x == num after t operations, the maximum initial x can be num + 2*t.
        return num + 2 * t