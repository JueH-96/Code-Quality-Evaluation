from typing import List
import math

class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)
        INF = math.inf
        
        # dp[i] will be the minimal total cost after segmenting the first i elements.
        dp = [INF]*(n+1)
        dp[0] = 0
        
        # For each segment (target value) we perform a pass over the array.
        for seg in range(m):
            target = andValues[seg]
            new_dp = [INF]*(n+1)
            # cur: dictionary representing the current segment being built.
            # key: accumulated AND value.
            # value: a tuple (base, cost) 
            #    where 'base' = dp value from the index where this segment started,
            #          and 'cost' = base + (last element of current segment).
            cur = {}
            # Process positions i = 0...n-1.
            for i in range(n):
                new_cur = {}
                # (1) If we can start a new segment at index i 
                #     (i.e. if dp[i] is computed from previous segmentation)
                if dp[i] != INF:
                    # starting a new segment from i: the candidate subarray is just [nums[i]]
                    mask = nums[i]
                    base = dp[i]
                    cost_val = base + nums[i]  # if we finish the subarray at index i.
                    # update new_cur for key mask:
                    if mask not in new_cur or cost_val < new_cur[mask][1]:
                        new_cur[mask] = (base, cost_val)
                # (2) Extend all previously started subarray (state) that ended at i-1
                for mask, (base, cost_val) in cur.items():
                    new_mask = mask & nums[i]
                    new_cost = base + nums[i]  # cost is always dp[start] + (last element) – note we replace the old last element cost.
                    if new_mask not in new_cur or new_cost < new_cur[new_mask][1]:
                        new_cur[new_mask] = (base, new_cost)
                # (3) If any state in new_cur has accumulated AND == target,
                # then we can “finish” the subarray at index i.
                for mask, (base, cost_val) in new_cur.items():
                    if mask == target:
                        # finishing the segment at index i, next segment will start at index i+1.
                        if cost_val < new_dp[i+1]:
                            new_dp[i+1] = cost_val
                # (4) Set cur for the next iteration (i+1) equal to new_cur.
                cur = new_cur
            # done with current segmentation pass: update dp to new_dp
            dp = new_dp
        
        # answer is dp at index n (i.e. having used all elements)
        return dp[n] if dp[n] != INF else -1


# For testing the solution with provided examples.
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    nums = [1,4,3,3,2]
    andValues = [0,3,3,2]
    print(sol.minimumValueSum(nums, andValues))  # Expected output: 12

    # Example 2:
    nums = [2,3,5,7,7,7,5]
    andValues = [0,7,5]
    print(sol.minimumValueSum(nums, andValues))  # Expected output: 17

    # Example 3:
    nums = [1,2,3,4]
    andValues = [2]
    print(sol.minimumValueSum(nums, andValues))  # Expected output: -1