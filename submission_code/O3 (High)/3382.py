from typing import List
import bisect
import collections

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        n = len(nums)

        # 1. previous index whose value is strictly greater than nums[i]
        prev_greater = [-1] * n
        stack = []                               # monotonically decreasing stack
        for i, v in enumerate(nums):
            while stack and nums[stack[-1]] <= v:
                stack.pop()
            prev_greater[i] = stack[-1] if stack else -1
            stack.append(i)

        # 2. sweep from left to right and count valid left-endpoints for every right endpoint
        positions = collections.defaultdict(list)   # value -> list of occurred indices
        ans = 0

        for i, v in enumerate(nums):
            pg = prev_greater[i]                    # barrier position for value v
            idx_list = positions[v]                 # previous occurrences of v (sorted)

            # number of indices > pg  (i.e. located after the nearest greater element)
            cnt = len(idx_list) - bisect.bisect_right(idx_list, pg)

            # +1 for the subarray consisting of the single element nums[i]
            ans += cnt + 1

            idx_list.append(i)                      # record current position

        return ans