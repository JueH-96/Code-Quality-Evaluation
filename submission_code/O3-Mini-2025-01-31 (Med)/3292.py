from typing import List

class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        n = len(nums)
        m = len(changeIndices)
        total_decrements = sum(nums)
        
        # The idea:
        # We have m seconds. In each second we choose one operation: either
        # (1) a decrement (affecting one chosen index) or 
        # (2) a mark operation if the designated index (given by changeIndices for that second)
        #     is currently 0.
        # We must eventually mark every index from 1..n (remember that nums is 1-indexed in the
        # problem statement, while changeIndices holds 1-indexed values).
        #
        # An optimal strategy is to “spend” as many seconds as possible doing decrement operations so that
        # later we can mark indices with more time having been allocated for decrements.
        #
        # Note that:
        #   • The total number of decrement operations needed (across all indices) is total_decrements = sum(nums).
        #   • Since exactly n seconds (one per index) must be used to mark indices,
        #     there are exactly s - n available seconds for decrements if we finish the process at second s.
        #     Hence a necessary condition is:
        #         s - n >= total_decrements.
        #
        # For each index i (1-indexed), we are allowed to mark it only at those seconds t in [1, s]
        # where changeIndices[t-1] equals i.
        # Furthermore, if we mark index i at second t we must have performed at least nums[i-1]
        # (its required number of decrements) earlier (i.e. in seconds before t).
        #
        # In an optimal schedule we can think of “reserving” a marking opportunity as late as possible.
        # For each index i, let best[i] be the largest second (between 1 and s) where changeIndices
        # equals i. (If no such second exists then marking index i is impossible.)
        #
        # Now suppose we assign to each index i the marking second best[i].
        # In any schedule the marking operations occur in increasing time order.
        # Imagine that when the j-th mark happens (j = 1, 2, …, n) at time T, there have been exactly (T - j)
        # seconds available to perform decrements (because j seconds are “used” by marking).
        # Hence, if an index with required decrement count req is to be marked at time T and it is the j-th mark,
        # a necessary condition is that T - j >= req.
        #
        # Thus, after sorting the indices by their best marking second, the j-th (1-indexed) mark must satisfy:
        #       best - j >= nums[i-1]
        #
        # Checking this (together with the global condition s - n >= total_decrements) is both necessary and,
        # as one may argue by a greedy scheduling argument, sufficient for feasibility.
        #
        # We then binary‐search on s (from 1 to m) for the smallest s that is feasible.
        
        def feasible(s: int) -> bool:
            # Global condition: we must have enough seconds for all decrements.
            if s - n < total_decrements:
                return False
            
            # For each index i (1-indexed), record the last (largest) time among the first s seconds
            # when changeIndices has value i. (0 will indicate that no opportunity exists.)
            best = [0] * (n + 1)
            for t in range(1, s + 1):
                idx = changeIndices[t-1]  # changeIndices gives indices 1-indexed
                if t > best[idx]:
                    best[idx] = t
            
            # Every index must have at least one marking opportunity.
            for i in range(1, n + 1):
                if best[i] == 0:
                    return False
            
            # Now for each index i, we have the requirement that its mark operation (if taken as late as possible,
            # i.e. at time best[i]) can only make use of best[i] - j decrements if it is the j-th marking.
            # Collect (best marking time, required decrements) pairs.
            pairs = []
            for i in range(1, n + 1):
                pairs.append((best[i], nums[i - 1]))
            pairs.sort(key=lambda x: x[0])
            
            j = 1
            for b, req in pairs:
                # When marking at time b as the j-th mark, the maximum decrements that can be done before b is b - j.
                if b - j < req:
                    return False
                j += 1
                
            return True
        
        # We can immediately rule out cases where there aren’t even enough seconds to allocate all required operations.
        # We need s - n >= total_decrements so that s must be at least total_decrements + n.
        low = total_decrements + n
        # Also s must be between 1 and m. (Recall m is the total seconds available.)
        if low > m:
            return -1
        
        ans = -1
        lo, hi = low, m
        while lo <= hi:
            mid = (lo + hi) // 2
            if feasible(mid):
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return ans

# --- Sample testing ---
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    # nums = [2,2,0], changeIndices = [2,2,2,2,3,2,2,1]
    # Expected output: 8
    print(sol.earliestSecondToMarkIndices([2,2,0], [2,2,2,2,3,2,2,1]))
    
    # Example 2:
    # nums = [1,3], changeIndices = [1,1,1,2,1,1,1]
    # Expected output: 6
    print(sol.earliestSecondToMarkIndices([1,3], [1,1,1,2,1,1,1]))
    
    # Example 3:
    # nums = [0,1], changeIndices = [2,2,2]
    # Expected output: -1 (because index 1 never appears in changeIndices)
    print(sol.earliestSecondToMarkIndices([0,1], [2,2,2]))