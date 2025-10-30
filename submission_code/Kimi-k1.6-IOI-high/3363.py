import heapq
from collections import defaultdict
from typing import List

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        freq_map = defaultdict(int)
        max_heap = []
        ans = []
        for i in range(len(nums)):
            id = nums[i]
            delta = freq[i]
            # Update the frequency of the current ID
            freq_map[id] += delta
            # Push the new frequency (as negative) into the heap
            heapq.heappush(max_heap, (-freq_map[id], id))
            # Find the current maximum valid frequency
            while max_heap:
                current_neg_count, current_id = max_heap[0]
                current_count = -current_neg_count
                if freq_map[current_id] == current_count:
                    ans.append(current_count)
                    break
                else:
                    # The top element is outdated, remove it
                    heapq.heappop(max_heap)
            else:
                # Heap is empty, no elements left
                ans.append(0)
        return ans