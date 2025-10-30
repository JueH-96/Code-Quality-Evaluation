class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # For each subsequence of nums, count how many of its subsequences sum to k
        # dp[i][s] = number of subsequences using first i elements with sum s
        # For each such subsequence, we need to count how many of its sub-subsequences sum to k
        
        # Let's think differently: 
        # result[i][s] = sum of powers of all subsequences using first i elements with sum s
        
        # dp[s] = (number of subsequences with sum s, sum of their powers)
        dp = {}
        dp[0] = (1, 1 if k == 0 else 0)  # empty set has 1 subsequence (itself), power is 1 if k==0
        
        for num in nums:
            new_dp = {}
            
            # Don't include current number
            for s, (count, power) in dp.items():
                if s not in new_dp:
                    new_dp[s] = (0, 0)
                new_dp[s] = ((new_dp[s][0] + count) % MOD, 
                            (new_dp[s][1] + power) % MOD)
            
            # Include current number
            for s, (count, power) in dp.items():
                new_s = s + num
                if new_s not in new_dp:
                    new_dp[new_s] = (0, 0)
                
                # Each subsequence with sum s becomes a subsequence with sum new_s
                # Its power increases by 1 if new_s == k (because we add one more way to get sum k)
                new_power = power
                if new_s == k:
                    new_power = (new_power + count) % MOD
                
                new_dp[new_s] = ((new_dp[new_s][0] + count) % MOD,
                                (new_dp[new_s][1] + new_power) % MOD)
            
            dp = new_dp
        
        total_power = 0
        for s, (count, power) in dp.items():
            total_power = (total_power + power) % MOD
        
        return total_power