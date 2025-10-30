from typing import List
import heapq

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        # Calculate the initial sum of nums1
        total_sum = sum(nums1)
        
        # If the initial sum is already less than or equal to x, return 0
        if total_sum <= x:
            return 0
        
        # Create a max heap to store the potential of each index to reduce the sum
        # Potential is calculated as (nums1[i] + nums2[i] * (time - 1)) - nums1[i]
        # which simplifies to nums2[i] * (time - 1)
        max_heap = [(-nums2[i], i) for i in range(len(nums1))]
        heapq.heapify(max_heap)
        
        time = 0
        while total_sum > x:
            time += 1
            # Get the index with the maximum potential to reduce the sum
            potential, i = heapq.heappop(max_heap)
            potential = -potential
            
            # Reduce the sum by setting nums1[i] to 0
            total_sum -= (nums1[i] + nums2[i] * (time - 1))
            nums1[i] = 0
            
            # If the sum is reduced enough, return the time
            if total_sum <= x:
                return time
            
            # Update the potential for the next time and push it back to the heap
            potential = nums2[i] * time
            heapq.heappush(max_heap, (-potential, i))
        
        # If we cannot reduce the sum to x, return -1
        return -1