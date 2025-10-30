from collections import Counter
import heapq

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        count = Counter(nums)
        heap = []
        for num, freq in count.items():
            heapq.heappush(heap, (num, freq))
        while len(heap) > 1:
            _, freq1 = heapq.heappop(heap)
            _, freq2 = heapq.heappop(heap)
            if freq1 > freq2:
                heapq.heappush(heap, (freq1-freq2, freq2))
        return len(heap)