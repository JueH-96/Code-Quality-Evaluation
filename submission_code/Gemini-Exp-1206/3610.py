from collections import Counter
from typing import List

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        n = len(nums)
        answer = []
        for i in range(n - k + 1):
            subarray = nums[i:i + k]
            if len(set(subarray)) <= x:
                answer.append(sum(subarray))
            else:
                count = Counter(subarray)
                sorted_count = sorted(count.items(), key=lambda item: (item[1], item[0]), reverse=True)
                top_x = sorted_count[:x]
                x_sum = 0
                for num, freq in top_x:
                    x_sum += num * freq
                answer.append(x_sum)
        return answer