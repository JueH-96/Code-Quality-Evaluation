from typing import List
import collections

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        id_count = collections.defaultdict(int)
        max_heap = []
        ans = []
        import heapq
        
        for i in range(len(nums)):
            id_count[nums[i]] += freq[i]
            
            # Push the current ID and its frequency into the heap
            heapq.heappush(max_heap, (-id_count[nums[i]], nums[i]))
            
            # Ensure the top of the heap is the most frequent ID
            while max_heap and id_count[max_heap[0][1]] != -max_heap[0][0]:
                heapq.heappop(max_heap)
            
            # The most frequent ID's frequency is the negative of the top element's frequency
            ans.append(-max_heap[0][0] if max_heap else 0)
        
        return ans