class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        import sys
        input_data = sys.stdin.readline
        mod = 10**9 + 7
        
        n = len(nums)
        
        # Precompute powers of 2 up to n (for multiplying by 2^(n-l))
        pow2 = [1]*(n+1)
        for i in range(1, n+1):
            pow2[i] = (pow2[i-1] * 2) % mod
        
        # dp[s][l] = number of subsequences (considering some prefix of nums)
        # that have sum = s and length = l
        dp = [[0]*(n+1) for _ in range(k+1)]
        dp[0][0] = 1  # There's exactly 1 way (the empty subsequence) to get sum=0, length=0
        
        for x in nums:
            new_dp = [[0]*(n+1) for _ in range(k+1)]
            for s in range(k+1):
                for length in range(n+1):
                    if dp[s][length] == 0:
                        continue
                    count_here = dp[s][length]
                    
                    # Option 1: do not include x
                    new_dp[s][length] = (new_dp[s][length] + count_here) % mod
                    
                    # Option 2: include x, if it doesn't exceed k
                    s_new = s + x
                    if s_new <= k and length+1 <= n:
                        new_dp[s_new][length+1] = (new_dp[s_new][length+1] + count_here) % mod
            
            dp = new_dp
        
        # Sum over all lengths l of dp[k][l] * 2^(n-l)
        ans = 0
        for l in range(n+1):
            if dp[k][l] != 0:
                ans = (ans + dp[k][l] * pow2[n - l]) % mod
        
        return ans