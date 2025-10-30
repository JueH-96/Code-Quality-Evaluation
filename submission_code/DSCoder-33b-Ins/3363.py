from typing import List
from collections import defaultdict
import heapq

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        id_freq = defaultdict(int)
        max_heap = []
        ans = []
        max_freq = 0

        for i in range(len(nums)):
            id_freq[nums[i]] += freq[i]
            heapq.heappush(max_heap, (-id_freq[nums[i]], -i, nums[i]))

            while max_heap and id_freq[max_heap[0][2]] != -max_heap[0][0]:
                heapq.heappop(max_heap)

            if max_heap:
                max_freq = -max_heap[0][0]

            ans.append(max_freq)

        return ans