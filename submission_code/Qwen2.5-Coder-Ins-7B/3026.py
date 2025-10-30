class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        if n >= target // 2:
            return (n * (n + 1)) // 2 + (target // 2) * (target - target // 2) // 2
        else:
            return (n * (n + 1)) // 2