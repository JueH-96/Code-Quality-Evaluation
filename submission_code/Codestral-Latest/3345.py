class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        power_sum = 0

        # Iterate through all possible subsequences
        for i in range(1 << n):
            subsequence_sum = 0
            for j in range(n):
                if i & (1 << j):
                    subsequence_sum += nums[j]
            if subsequence_sum == k:
                power_sum += 1

        return power_sum % MOD