from collections import Counter
import heapq

class Solution:
    def maxFrequencyScore(self, nums: List[int], k: int) -> int:
        counter = Counter(nums)
        max_heap = []
        for num, freq in counter.items():
            heapq.heappush(max_heap, (-freq, num))
        
        while k > 0:
            freq, num = heapq.heappop(max_heap)
            if k >= num:
                k -= num
                freq += 1
                heapq.heappush(max_heap, (freq, num))
            else:
                heapq.heappush(max_heap, (freq, num))
                break
        
        return -max_heap[0][0]