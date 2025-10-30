class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        seen = set()
        nums = []
        curr = 1
        for _ in range(n):
            while curr in seen or (target - curr in seen and target - curr != curr):
                curr += 1
            nums.append(curr)
            seen.add(curr)
        return sum(nums) % MOD