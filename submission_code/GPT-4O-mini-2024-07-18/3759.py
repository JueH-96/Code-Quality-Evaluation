from typing import List
import heapq

class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        answer = [0] * n
        
        # Create a list of tuples (value from nums1, index) and sort it
        indexed_nums1 = sorted((value, index) for index, value in enumerate(nums1))
        
        # This will store the max heap of values from nums2
        max_heap = []
        
        # Iterate through each index in the sorted nums1
        for i in range(n):
            current_value = indexed_nums1[i][0]
            current_index = indexed_nums1[i][1]
            
            # Add all valid nums2[j] to the max_heap where nums1[j] < current_value
            while max_heap and nums1[max_heap[0][1]] < current_value:
                heapq.heappush(max_heap, (nums2[max_heap[0][1]], max_heap[0][1]))
                heapq.heappop(max_heap)
            
            # We need to take the largest k values from the heap
            if len(max_heap) > k:
                largest_k = heapq.nlargest(k, max_heap)
                answer[current_index] = sum(value for value, _ in largest_k)
            else:
                answer[current_index] = sum(value for value, _ in max_heap)
        
        return answer