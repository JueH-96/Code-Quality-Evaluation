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
                # Since we can't efficiently remove from a heap, we'll mark it as invalid
                # and push the new count
                pass
            id_counts[num] += f
            if id_counts[num] == 0:
                del id_counts[num]
            # Push the new count to the heap
            heapq.heappush(max_heap, (-id_counts[num], num))
            # Now, we need to ensure that the top of the heap is valid
            while max_heap and id_counts[max_heap[0][1]] != -max_heap[0][0]:
                heapq.heappop(max_heap)
            if max_heap:
                ans.append(-max_heap[0][0])
            else:
                ans.append(0)
        
        return ans