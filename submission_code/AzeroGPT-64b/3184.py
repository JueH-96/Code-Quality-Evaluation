from typing import List
from sortedcontainers import SortedList

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        max_sum = -1 * (10**9 + 1)
        num_map = SortedList()

        for i, num in enumerate(nums):
            val_index = num - i

            if num_map.bisect_right(val_index) > 0:
                max_index_candidate = num_map.bisect_right(val_index) - 1
                prev_sum = num_map[max_index_candidate][1]
                new_sum = max(prev_sum + num, num)
                prev_number, prev_max = num_map[max_index_candidate]

            else:
                new_sum = num
                prev_number = prev_max = -1

            # Replace the element if possible or insert it at the correct position
            if prev_number <= val_index:
                if num_map.bisect_right(val_index) > 0:
                    num_map[num_map.bisect_right(val_index) - 1] = (val_index, new_sum)

                elif num_map.bisect_right(val_index) == len(num_map):
                    num_map.add((val_index, new_sum))
            else:
                num_map.add((val_index, new_sum))

            max_sum = max(max_sum, new_sum)

        return max_sum