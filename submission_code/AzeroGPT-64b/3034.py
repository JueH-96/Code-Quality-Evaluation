from typing import List

class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        covered_points = set()
        for start, end in nums:
            for point in range(start, end + 1):
                covered_points.add(point)
        return len(covered_points)

        # Alternative method using intervals merging technique
        # intervals = sorted(nums)
        # merged_intervals = [intervals[0]]
        # for current_start, current_end in intervals[1:]:
        #     last_start, last_end = merged_intervals[-1]
        #     if current_start <= last_end + 1:
        #         merged_intervals[-1] = (last_start, max(last_end, current_end))
        #     else:
        #         merged_intervals.append((current_start, current_end))
        # return sum(end - start + 1 for start, end in merged_intervals)