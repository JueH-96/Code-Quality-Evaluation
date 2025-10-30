from typing import List
import heapq

class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        answer = [0] * n
        
        for i in range(n):
            # Find all indices j where nums1[j] < nums1[i]
            indices = [j for j in range(n) if nums1[j] < nums1[i]]
            
            # Get the corresponding nums2 values
            values = [nums2[j] for j in indices]
            
            # If there are values, find the k largest
            if values:
                # Use a min-heap to find the k largest values
                largest_values = heapq.nlargest(k, values)
                answer[i] = sum(largest_values)
            else:
                answer[i] = 0
        
        return answer