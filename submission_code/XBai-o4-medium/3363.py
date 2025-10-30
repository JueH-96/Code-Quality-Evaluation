import heapq
from typing import List

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        counts = dict()
        heap = []
        ans = []
        for i in range(len(nums)):
            num = nums[i]
            delta = freq[i]
            prev = counts.get(num, 0)
            new_count = prev + delta
            # Update the counts dictionary
            if new_count == 0:
                if num in counts:
                    del counts[num]
            else:
                counts[num] = new_count
            # Push to heap if new_count is not zero
            if new_count != 0:
                heapq.heappush(heap, (-new_count, num))
            # Process the heap to remove invalid entries
            while heap:
                current_count_neg, current_id = heap[0]
                current_count = -current_count_neg
                if counts.get(current_id, 0) == current_count:
                    break
                else:
                    heapq.heappop(heap)
            # Determine the current maximum frequency
            if heap:
                ans.append(-heap[0][0])
            else:
                ans.append(0)
        return ans