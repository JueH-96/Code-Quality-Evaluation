import heapq
from typing import List

class Solution:
    def findMaxSum(self, nums1: List[int], nums2: List[int], k: int) -> List[int]:
        n = len(nums1)
        elements = sorted([(nums1[i], nums2[i], i) for i in range(n)], key=lambda x: x[0])
        answer = [0] * n
        heap = []
        current_sum = 0
        current_group = []
        prev_num = None

        for elem in elements:
            num, val, idx = elem
            if num != prev_num:
                # Process the previous group
                if current_group:
                    # Set answers for all elements in current_group
                    for e in current_group:
                        answer[e[2]] = current_sum
                    # Add all elements in current_group to the heap
                    for e in current_group:
                        v = e[1]
                        if len(heap) < k:
                            heapq.heappush(heap, v)
                            current_sum += v
                        else:
                            if v > heap[0]:
                                popped = heapq.heappop(heap)
                                current_sum -= popped
                                heapq.heappush(heap, v)
                                current_sum += v
                # Start new group
                current_group = [elem]
                prev_num = num
            else:
                current_group.append(elem)
        
        # Process the last group
        if current_group:
            for e in current_group:
                answer[e[2]] = current_sum
            # Adding the last group to the heap is not needed for the answer but required for correctness
            for e in current_group:
                v = e[1]
                if len(heap) < k:
                    heapq.heappush(heap, v)
                    current_sum += v
                else:
                    if v > heap[0]:
                        popped = heapq.heappop(heap)
                        current_sum -= popped
                        heapq.heappush(heap, v)
                        current_sum += v
        
        return answer