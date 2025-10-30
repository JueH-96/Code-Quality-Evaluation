from collections import Counter
from sortedcontainers import SortedList

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        n = len(nums)
        sorted_nums = SortedList()
        counter = Counter()
        score = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            sorted_nums.add(nums[i])
            counter[nums[i]] += 1
            score[i] = sorted_nums[0] * counter[sorted_nums[0]]
            if sorted_nums:
                sorted_nums.remove(nums[i])
                counter[nums[i]] -= 1
        return score[0]