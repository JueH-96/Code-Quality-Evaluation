from typing import List
import math

class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)
        
        # dp[i][j] will represent the minimum sum we can get by partitioning nums[0..i]
        # into j continuous subarrays satisfying conditions, where j ranges 1..m.
        # We use a table dp[i][j] and use math.inf for unreachable states.
        dp = [[math.inf] * (m + 1) for _ in range(n)]
        
        # For each subarray ending at index i, try to form one segment.
        # j is the number of segments used.
        # We fill dp[i][j] where j>=1.
        for i in range(n):
            # We'll try to fill dp[i][j] for each possible j (the last segment ending at i).
            # We compute the AND value for the subarray ending at i by expanding backwards.
            curr_and = (1 << 30) - 1  # set all bits  (or you could use ~0, but this is safe for our range)
            # Because constraints: 0 <= nums[i], andValues < 10^5, so 30 bits is safe.
            for k in range(i, -1, -1):
                curr_and &= nums[k]
                # if this subarray [k, i] produces the required AND for some segment index j,
                # we are allowed to use it only if curr_and equals andValues[j-1] (if we are trying to form the j-th segment).
                # But we also add the value of the segment, which is defined as nums[i], the last element of the subarray.
                # We can break early if it becomes impossible to match the target.
                # Check bitmask feasibility: For a target T, we must have (curr_and & T == T) otherwise adding more numbers
                # cannot bring bits back up.
                # However, note that even if this condition holds, we need equality at the segment boundary.
                # So we check equality.
                
                # For segment index 1 (first segment), k must be 0.
                # For segment index >1, we need a valid previous partition ending at k-1.
                # We'll iterate over possible segment counts in reverse order.
                
                # Try for segment counts from m down to 1 for which we are forming the last segment ending at i.
                # Actually, we can only use one segment count at a time based on our dp transition.
                # So we update dp[i][seg_count] if the current segment's AND equals andValues[seg_count-1].
                # We want to combine with best dp[k-1][seg_count-1].
                # We iterate only for the specific segment count that might end at i; that is, if current AND equals andValues[seg_count-1].
                
                # But here we traverse each k as possible start for the last segment.
                # Instead, we check: if curr_and equals a candidate target for a segment.
                
                # For subarray [k, i], if curr_and equals some target, then if k==0, that would form the 1st segment; else combine with previous dp.
                # We try: for j = 1 to m, but the j for this segment must satisfy: target = andValues[j-1].
                # So if curr_and is exactly one of the required values, then we try that segment.
                
                # However, note that the segments are in order. When processing a candidate ending at i,
                # the segment we are forming must be the j-th in order.
                # So we try: if curr_and == andValues[j-1], then:
                #   if j==1 and k==0, dp[i][1] = min(dp[i][1], nums[i])
                #   if j > 1 and k > 0 and dp[k-1][j-1] != inf, then dp[i][j] = min(dp[i][j], dp[k-1][j-1] + nums[i])
                
                # We can try potential j where this candidate segment fits.
                for seg in range(1, m + 1):
                    # Only if the required AND for seg-th segment matches
                    if curr_and == andValues[seg - 1]:
                        if seg == 1:
                            if k == 0:  # Only valid if the subarray starts at index 0.
                                dp[i][1] = min(dp[i][1], nums[i])
                        else:
                            if k > 0 and dp[k - 1][seg - 1] != math.inf:
                                dp[i][seg] = min(dp[i][seg], dp[k - 1][seg - 1] + nums[i])
                # Early break:
                # If curr_and no longer can possibly match any future required target of a segment (for example,
                # if curr_and & andValues[j-1] != andValues[j-1] for all j where the target has bits that curr_and lost),
                # we can break out early. However, m is small so it's not a huge optimization.
                # We can check if curr_and == 0 then further extension can only remain 0.
                # But even that might be the target in some cases, so we cannot break unconditionally.
                # Instead, we check: if curr_and == 0 and 0 is not present in any andValues, then break.
                # However, due to small m, skipping this optimization.
                
        # After filling dp, the answer is the minimum value among dp[i][m] for i from m-1 to n-1.
        ans = math.inf
        for i in range(m - 1, n):
            ans = min(ans, dp[i][m])
        return ans if ans != math.inf else -1