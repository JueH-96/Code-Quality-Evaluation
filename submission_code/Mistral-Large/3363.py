from typing import List
from collections import defaultdict
import heapq

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        count = defaultdict(int)
        max_heap = []
        result = []

        for i in range(len(nums)):
            num = nums[i]
            f = freq[i]

            if count[num] == 0:
                heapq.heappush(max_heap, (-f, num))
            else:
                for idx, (neg_count, id) in enumerate(max_heap):
                    if id == num:
                        max_heap[idx] = (-(count[num] + f), num)
                        break

            count[num] += f

            while max_heap and count[max_heap[0][1]] == 0:
                heapq.heappop(max_heap)

            if max_heap:
                result.append(-max_heap[0][0])
            else:
                result.append(0)

        return result