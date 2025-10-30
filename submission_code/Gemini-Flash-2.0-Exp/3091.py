class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        mod = 10**9 + 7
        count_zero = nums.count(0)
        nums = [num for num in nums if num != 0]
        max_sum = sum(nums)
        if max_sum == 0:
            if 0 >= l and 0 <= r:
                return pow(2, len(nums) + count_zero, mod)
            else:
                return 0

        dp = [0] * (max_sum + 1)
        dp[0] = 1
        for num in nums:
            for i in range(num, max_sum + 1):
                dp[i] = (dp[i] + dp[i - num]) % mod
        
        ans = 0
        for i in range(l, r + 1):
            ans = (ans + dp[i]) % mod
        
        ans = (ans * pow(count_zero + 1, 1, mod)) % mod
        return ans