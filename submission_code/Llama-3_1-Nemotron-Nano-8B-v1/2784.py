class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        if n == 0:
            return 0
        total = 0
        prev_S = 0
        for j in range(n):
            current = nums[j]
            if j == 0:
                S_j = 0
            else:
                S_j = (2 * prev_S + nums[j-1]) % MOD
            contrib = (pow(current, 3, MOD) + (pow(current, 2, MOD) * S_j) % MOD) % MOD
            total = (total + contrib) % MOD
            if j > 0:
                prev_S = S_j
        return total % MOD