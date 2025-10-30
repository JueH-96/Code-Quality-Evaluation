class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        prev_dp = 0
        total = 0
        for x in nums:
            current_s = (x + prev_dp) % MOD
            contrib = (current_s * x) % MOD
            contrib = (contrib * x) % MOD
            total = (total + contrib) % MOD
            prev_dp = ( (prev_dp * 2) + x ) % MOD
        return total