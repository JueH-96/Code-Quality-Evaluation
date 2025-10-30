from collections import Counter
from typing import List

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        def x_sum(subarray):
            counter = Counter(subarray)
            most_common = counter.most_common()
            most_common.sort(key=lambda x: (-x[1], -x[0]))
            top_x = most_common[:x]
            return sum(num * count for num, count in top_x)

        result = []
        for i in range(len(nums) - k + 1):
            subarray = nums[i:i + k]
            result.append(x_sum(subarray))

        return result