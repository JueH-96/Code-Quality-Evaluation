import collections

class Solution:
    def countSubMultisets(self, nums: list[int], l: int, r: int) -> int:
        """
        Calculates the number of sub-multisets of `nums` with a sum in the range [l, r].

        The method uses dynamic programming. Let dp[s] be the number of sub-multisets
        with sum s. We build this dp table by iteratively considering each unique
        number and its frequency from the input `nums`.

        1.  Frequency Counting: First, we count the occurrences of each number in `nums`.
            Zeros are handled specially. A sub-multiset of `c0` zeros can be chosen in
            `c0 + 1` ways. This acts as a multiplier to the final result based on
            non-zero elements.

        2.  Dynamic Programming with Generating Functions: For non-zero elements, we
            process one unique value `v` with its count `c` at a time. This is a
            bounded knapsack problem. The update can be efficiently performed using a
            recurrence derived from generating functions:
                dp_new[s] = dp_new[s-v] + dp_old[s] - dp_old[s-(c+1)*v]
            This recurrence allows updating the dp table in O(r) for each unique item.

        3.  Complexity Analysis: A key insight is that the number of distinct non-zero
            elements (`D`) is small (at most ~200) due to the constraint on the total
            sum of `nums`. This makes the overall complexity of the DP part O(D * r),
            which is efficient enough.

        4.  Final Calculation: After processing all unique non-zero numbers, the `dp`
            table holds the counts for sub-multisets of non-zero elements. We sum
            the counts for sums in `[l, r]` and multiply by `(c0 + 1)` for the final answer.
        """
        MOD = 10**9 + 7
        
        counts = collections.Counter(nums)
        
        c0 = counts.pop(0, 0)
        
        dp = [0] * (r + 1)
        dp[0] = 1
        
        for v, c in counts.items():
            prev_dp = dp
            dp = [0] * (r + 1)
            
            for s in range(r + 1):
                # Recurrence: dp_new[s] = dp_new[s-v] + dp_old[s] - dp_old[s-(c+1)v]
                
                # Contribution from using one more item `v`, based on the new dp table
                term_from_new = dp[s - v] if s >= v else 0
                
                # Contribution from the previous state (before considering item `v`)
                term_from_old = prev_dp[s]
                
                # To enforce the bound of `c` items, we subtract cases where we would
                # have used more than `c` items. This is equivalent to removing ways
                # to form sum s-(c+1)*v from the unbounded case.
                removal_offset = (c + 1) * v
                removal_term = prev_dp[s - removal_offset] if s >= removal_offset else 0
                
                dp[s] = (term_from_new + term_from_old - removal_term + MOD) % MOD
                
        total_count = sum(dp[s] for s in range(l, r + 1)) % MOD
        
        return (total_count * (c0 + 1)) % MOD