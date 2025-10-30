from typing import List
import math

class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)
        INF = math.inf

        # dp is a dictionary mapping partition boundary index to minimal cost.
        # dp[i] = minimal cost to partition nums[0:i] into some segments.
        dp = {0: 0}  # initially, no elements partitioned: cost 0

        # Process each segment (each required andValue) one by one.
        for seg in range(m):
            target = andValues[seg]  # required AND for this segment
            new_dp = {}  # dp for partition boundaries after adding this segment
            # state dictionary holds, for the subarray currently being formed,
            # the current cumulative AND value and the minimal inherited cost 
            # from the dp that started that segment.
            state = {}
            # We iterate over indices of nums, considering ending positions for the current segment.
            for i in range(n):
                new_state = {}
                # Extend every candidate segment in state with nums[i]
                for val, cost in state.items():
                    new_val = val & nums[i]
                    if new_val in new_state:
                        if cost < new_state[new_val]:
                            new_state[new_val] = cost
                    else:
                        new_state[new_val] = cost
                # Also, if there is a partition ending exactly at i, then a new segment may start at i.
                if i in dp:
                    # When starting a new segment at index i, its accumulated AND becomes nums[i]
                    cand = dp[i]
                    new_val = nums[i]
                    if new_val in new_state:
                        if cand < new_state[new_val]:
                            new_state[new_val] = cand
                    else:
                        new_state[new_val] = cand
                state = new_state

                # If any active candidate segment (ending at i) has cumulative AND equal to target,
                # then we may complete the segment with cost = inherited cost + value of last element.
                if target in state:
                    candidate_cost = state[target] + nums[i]
                    # The partition boundary is then i+1.
                    if (i+1) in new_dp:
                        if candidate_cost < new_dp[i+1]:
                            new_dp[i+1] = candidate_cost
                    else:
                        new_dp[i+1] = candidate_cost
            # After processing all positions, new_dp holds valid boundaries for this complete segment.
            dp = new_dp
            # If dp is empty, then no valid segmentation for the current segment was found.
            if not dp:
                return -1

        # After processing all m segments, a valid partition must have covered the entire array,
        # i.e. the partition boundary n. If not, it's not possible.
        return dp.get(n, -1)