class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        # Since nums is non-empty by constraints, no need to check for empty.
        maxv = max(nums)
        # We use an offset of 1 so that v=0 maps to index 1,
        # and we can safely look at idx-1 and idx+1 without negative indices.
        size = maxv + 3
        offset = 1
        
        # dp_count[i] = number of good subsequences ending with value (i - offset)
        # dp_sum[i]   = total sum of elements over all good subsequences ending with (i - offset)
        dp_count = [0] * size
        dp_sum   = [0] * size
        
        ans = 0
        for v in nums:
            idx = v + offset
            # counts and sums for subsequences ending in v-1 and v+1
            c1, s1 = dp_count[idx-1], dp_sum[idx-1]
            c2, s2 = dp_count[idx+1], dp_sum[idx+1]
            
            # We can start a new subsequence [v] (count=1, sum=v),
            # or extend those ending in v-1 or v+1.
            ext_count = (c1 + c2) % MOD
            new_count = (1 + ext_count) % MOD
            # sum of new subsequences: 
            #   - the singleton [v] contributes v
            #   - extending subseqs ending in v-1 contributes s1 + c1*v
            #   - extending subseqs ending in v+1 contributes s2 + c2*v
            new_sum = (v + s1 + s2 + v * ext_count) % MOD
            
            # accumulate into our DP tables
            dp_count[idx] = (dp_count[idx] + new_count) % MOD
            dp_sum[idx]   = (dp_sum[idx]   + new_sum)   % MOD
            
            # these new subsequences contribute to the final answer
            ans = (ans + new_sum) % MOD
        
        return ans