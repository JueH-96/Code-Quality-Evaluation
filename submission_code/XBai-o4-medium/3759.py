from typing import List
import heapq

class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        # Create a list of tuples (nums1[i], nums2[i], original_index) sorted by nums1
        sorted_list = sorted([(nums1[i], nums2[i], i) for i in range(n)], key=lambda x: x[0])
        
        answer = [0] * n
        heap = []
        current_sum = 0
        
        i = 0
        while i < n:
            current_group_val = sorted_list[i][0]
            current_group = []
            # Collect all elements in the current group
            while i < n and sorted_list[i][0] == current_group_val:
                current_group.append(sorted_list[i])
                i += 1
            # Set the answer for each element in the current group
            for elem in current_group:
                answer[elem[2]] = current_sum
            # Update the heap with the current group's nums2 values
            for elem in current_group:
                val = elem[1]
                if len(heap) < k:
                    heapq.heappush(heap, val)
                    current_sum += val
                else:
                    if val > heap[0]:
                        popped = heapq.heappushpop(heap, val)
                        current_sum += (val - popped)
        return answer