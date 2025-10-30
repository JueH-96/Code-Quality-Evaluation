class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7
        nums = []
        i = 1
        while len(nums) < n:
            if target - i not in nums:
                nums.append(i)
            i += 1
        return sum(nums) % MOD