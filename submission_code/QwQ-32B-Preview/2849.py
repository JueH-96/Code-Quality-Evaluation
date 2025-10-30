from typing import List
import bisect

class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        total_imbalance = 0
        for l in range(n):
            sorted_list = [nums[l]]
            imbalance = 0
            for r in range(l + 1, n):
                insertion_point = bisect.bisect_left(sorted_list, nums[r])
                if insertion_point < len(sorted_list) and sorted_list[insertion_point] == nums[r]:
                    continue  # Duplicate, skip
                bisect.insort_left(sorted_list, nums[r])
                x = nums[r]
                left_neighbor = sorted_list[insertion_point - 1] if insertion_point > 0 else None
                right_neighbor = sorted_list[insertion_point + 1] if insertion_point < len(sorted_list) - 1 else None
                if left_neighbor is not None and x - left_neighbor > 1:
                    imbalance += 1
                if right_neighbor is not None and right_neighbor - x > 1:
                    imbalance += 1
                if left_neighbor is not None and right_neighbor is not None and right_neighbor - left_neighbor > 1:
                    imbalance -= 1
                total_imbalance += imbalance
        return total_imbalance