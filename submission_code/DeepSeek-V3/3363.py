from collections import defaultdict
import heapq

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        id_counts = defaultdict(int)
        max_heap = []
        ans = []
        
        for i in range(len(nums)):
            num = nums[i]
            f = freq[i]
            if id_counts[num] != 0:
                # Remove the old count from the heap
                # Since it's a max heap, we need to mark it as invalid
                # We can't directly remove elements from the heap, so we'll manage it differently
                # Instead, we'll keep track of the current counts and only consider the top of the heap if it matches the current count
                pass
            id_counts[num] += f
            if id_counts[num] == 0:
                del id_counts[num]
            # Push the new count to the heap
            heapq.heappush(max_heap, (-id_counts[num], num))
            # Now, we need to ensure that the top of the heap is valid
            while max_heap:
                current_max, current_num = max_heap[0]
                if -current_max == id_counts.get(current_num, 0):
                    break
                heapq.heappop(max_heap)
            if max_heap:
                ans.append(-max_heap[0][0])
            else:
                ans.append(0)
        
        return ans