class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        import sys
        from collections import Counter
        
        MOD = 10**9 + 7
        
        # Frequency map of the elements
        freq = Counter(nums)
        
        # Total sum of all elements in nums
        S = sum(nums)
        
        # dp[s] will represent the number of sub-multisets that have sum = s
        dp = [0] * (S + 1)
        dp[0] = 1  # The empty sub-multiset has sum 0
        
        # Process each distinct element with its frequency
        for val, count in freq.items():
            # Special quick handling for val == 0: it only affects sum=0
            if val == 0:
                # We can choose 0 up to count copies of 0, so that multiplies dp[0] by (count+1)
                dp[0] = (dp[0] * (count + 1)) % MOD
                continue
            
            # Build prefix sums of the current dp for O(1) range-sum retrieval
            prefix_dp = [0] * (S + 1)
            prefix_dp[0] = dp[0]
            for i in range(1, S + 1):
                prefix_dp[i] = (prefix_dp[i - 1] + dp[i]) % MOD
            
            new_dp = [0] * (S + 1)
            
            # For each possible sum j, compute how many ways if we use val up to 'count' times
            # Using the formula with prefix sums:
            # new_dp[j] = prefix_dp[j] - prefix_dp[j - (count+1)*val] (if j-(count+1)*val >= 0)
            for j in range(S + 1):
                left = j - (count + 1) * val
                if left >= 0:
                    new_dp[j] = (prefix_dp[j] - prefix_dp[left]) % MOD
                else:
                    new_dp[j] = prefix_dp[j] % MOD
            
            dp = new_dp
        
        # Now dp[s] = number of sub-multisets with sum = s
        # We want the sum of dp[s] for s in [l, r], clipped by [0, S]
        low = max(0, l)
        high = min(S, r)
        
        # Build final prefix sums on dp to quickly extract range sums
        final_prefix = [0] * (S + 1)
        final_prefix[0] = dp[0]
        for i in range(1, S + 1):
            final_prefix[i] = (final_prefix[i - 1] + dp[i]) % MOD
        
        if low > high:
            return 0  # no valid sums if the interval is out of range
        
        ans = final_prefix[high]
        if low > 0:
            ans = (ans - final_prefix[low - 1]) % MOD
        
        return ans % MOD