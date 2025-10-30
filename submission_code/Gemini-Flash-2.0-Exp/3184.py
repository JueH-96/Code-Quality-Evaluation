class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        arr = []
        for i in range(n):
            arr.append(nums[i] - i)
        
        sorted_arr = sorted(list(set(arr)))
        
        bit = {}
        
        def update(x, val):
            while x < len(sorted_arr) + 1:
                if x not in bit:
                    bit[x] = 0
                bit[x] = max(bit[x], val)
                x += x & -x
        
        def query(x):
            res = float('-inf')
            while x > 0:
                if x in bit:
                    res = max(res, bit[x])
                x -= x & -x
            return res
        
        dp = {}
        
        for i in range(n):
            val = nums[i] - i
            idx = sorted_arr.index(val) + 1
            
            prev_sum = query(idx)
            
            if prev_sum == float('-inf'):
                dp[i] = nums[i]
            else:
                dp[i] = prev_sum + nums[i]
            
            update(idx, dp[i])
        
        ans = float('-inf')
        for i in range(n):
            ans = max(ans, dp[i])
        
        return ans