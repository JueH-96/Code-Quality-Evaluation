from typing import List
import heapq

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        id_counts = {}
        max_heap = []
        result = []
        
        for i, (num, f) in enumerate(zip(nums, freq)):
            id_counts[num] = id_counts.get(num, 0) + f
            
            # Update the max heap
            while max_heap and id_counts[max_heap[0][1]] != -max_heap[0][0]:
                heapq.heappop(max_heap)
            heapq.heappush(max_heap, (-id_counts[num], num))
            
            # Find the most frequent ID count
            most_frequent_count = 0
            while max_heap and id_counts[max_heap[0][1]] != -max_heap[0][0]:
                heapq.heappop(max_heap)
            if max_heap:
                most_frequent_count = -max_heap[0][0]
            
            result.append(most_frequent_count)
        
        return result