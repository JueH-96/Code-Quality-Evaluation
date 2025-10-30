class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        nums.sort()
        n = len(nums)
        if n == 0:
            return 0
        
        prev_S = nums[0]
        total = (nums[0] ** 3) % MOD
        
        for i in range(1, n):
            a_i = nums[i]
            a_i_minus_1 = nums[i-1]
            current_S = (2 * prev_S - a_i_minus_1 + a_i) % MOD
            contribution = (a_i * a_i % MOD) * current_S % MOD
            total = (total + contribution) % MOD
            prev_S = current_S
        
        return total