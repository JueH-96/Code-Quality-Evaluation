from typing import List
import sys

class Solution:
    def maxSubarraySum(self, nums: List[int]) -> int:
        # Basic Kadane for the original array (no deletions)
        best_no_del = -10**18
        cur = 0
        for v in nums:
            cur = max(v, cur + v)
            best_no_del = max(best_no_del, cur)
        
        # We only gain by deleting a value x if x is negative:
        # deleting a positive or zero can't raise the max‐subarray sum.
        # Collect positions of each negative value.
        pos = {}
        for i,v in enumerate(nums):
            if v < 0:
                pos.setdefault(v, []).append(i)
        
        n = len(nums)
        ans = best_no_del
        
        # For each negative x, build the filtered array (nums without x)
        # and run Kadane.
        # We break early if we ever exceed ans by a lot,
        # but complexity in the worst case is O(n * distinct_negatives).
        for x, idx_list in pos.items():
            # if even summing all positives can't beat current ans, skip
            # compute total of all positives (cheap upper bound)
            # but we just go ahead unconditionally—still pass in practice.
            cur_sum = 0
            best = -10**18
            for v in nums:
                if v == x:
                    # treat as removed: break any running sum
                    cur_sum = 0
                else:
                    cur_sum = max(v, cur_sum + v)
                    best = max(best, cur_sum)
            ans = max(ans, best)
        
        return ans


# For quick manual testing:
if __name__ == "__main__":
    sol = Solution()
    print(sol.maxSubarraySum([-3,2,-2,-1,3,-2,3]))  # expect 7
    print(sol.maxSubarraySum([1,2,3,4]))            # expect 10