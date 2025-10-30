from typing import List

class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        sorted_nums = sorted(nums)
        current_list = sorted_nums
        averages = []
        while current_list:
            min_e = current_list[0]
            max_e = current_list[-1]
            avg = (min_e + max_e) / 2.0
            averages.append(avg)
            current_list = current_list[1:-1]
        return min(averages)