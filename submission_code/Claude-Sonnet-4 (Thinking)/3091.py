class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        
        from collections import Counter
        count = Counter(nums)
        
        # Handle zeros separately
        zero_count = count.get(0, 0)
        if 0 in count:
            del count[0]
        
        total_sum = sum(nums)
        
        # dp[s] = number of ways to achieve sum s using non-zero elements
        dp = [0] * (total_sum + 1)
        dp[0] = 1
        
        for val, cnt in count.items():
            new_dp = dp[:]
            for s in range(total_sum + 1):
                for k in range(1, cnt + 1):
                    if s + k * val <= total_sum:
                        new_dp[s + k * val] = (new_dp[s + k * val] + dp[s]) % MOD
            dp = new_dp
        
        # Account for zeros: each sum s can be achieved in dp[s] * (zero_count + 1) ways
        result = 0
        for s in range(l, min(r + 1, total_sum + 1)):
            result = (result + dp[s] * (zero_count + 1)) % MOD
        
        return result