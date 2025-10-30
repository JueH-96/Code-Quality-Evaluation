class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        import sys
        input_data = sys.stdin.readline  # Not strictly needed here, but often speeds up in some environments
        
        MOD = 10**9 + 7
        
        # Special handling if no valid range:
        if l > r:
            return 0
        
        from collections import Counter
        freq = Counter(nums)
        
        # Handle the count of zero separately, as choosing any number of zeros does not change the sum
        zero_count = freq.get(0, 0)
        if zero_count > 0:
            del freq[0]
        
        # If the entire array was zeros:
        # sum = 0 for any sub-multiset. 
        # So the only possible sum is 0, and the number of such sub-multisets is (zero_count+1).
        # If 0 is in [l, r], return (zero_count+1), else return 0.
        if len(freq) == 0:
            if l <= 0 <= r:
                return (zero_count + 1) % MOD
            else:
                return 0
        
        # We only need to compute DP up to 'r' because sums greater than r do not affect the final answer.
        max_sum = r
        
        # Initialize dp array: dp[s] = number of ways to form sum s using the processed distinct values so far
        dp = [0] * (max_sum + 1)
        dp[0] = 1  # There's exactly 1 way to form sum=0 (pick nothing from these processed values)
        
        # Process each distinct nonzero value (v) with its frequency (f)
        for v, f in freq.items():
            # Build prefix sums of dp
            prefix = [0] * (max_sum + 1)
            prefix[0] = dp[0] % MOD
            for i in range(1, max_sum + 1):
                prefix[i] = (prefix[i-1] + dp[i]) % MOD
            
            new_dp = [0] * (max_sum + 1)
            
            # new_dp[s] = sum of dp[s - k*v] for k=0..f, where s-k*v >= 0
            # Using prefix sums:
            # new_dp[s] = prefix[s] - prefix[s - (f+1)*v] (if that index is >= 0)
            for s in range(max_sum + 1):
                # min index below which we won't sum
                cutoff = s - (f+1)*v
                val = prefix[s]
                if cutoff >= 0:
                    val -= prefix[cutoff]
                new_dp[s] = val % MOD
            
            dp = new_dp
        
        # Now dp[s] = number of ways to form sum s from all nonzero elements.
        # Including zero_count zeros: we can choose from 0..zero_count zeros in any combination (which is (zero_count+1) ways),
        # so each valid sum s is multiplied by (zero_count+1).
        zero_factor = zero_count + 1
        
        ans = 0
        # Sum up dp[s]*(zero_factor) for s in [l..r], within bounds
        left_bound = max(l, 0)  # sums < 0 are impossible anyway, no negative sums
        right_bound = min(r, max_sum)
        for s in range(left_bound, right_bound + 1):
            ans = (ans + dp[s] * zero_factor) % MOD
        
        return ans % MOD