from typing import List
import heapq
from collections import defaultdict

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        n = len(nums)
        ans = []
        freq_dict = dict()  # Maps ID to its current count
        count_freq = defaultdict(int)  # Maps count to the number of IDs with that count
        heap = []  # Max-heap simulated with min-heap using negative counts
        
        for i in range(n):
            id_ = nums[i]
            change = freq[i]
            
            old_count = freq_dict.get(id_, 0)
            new_count = old_count + change
            
            # Update count_freq for the old count
            if old_count != 0:
                count_freq[old_count] -= 1
                if count_freq[old_count] == 0:
                    del count_freq[old_count]
            
            if new_count == 0:
                # Remove the ID from freq_dict if new_count is zero
                if id_ in freq_dict:
                    del freq_dict[id_]
            else:
                # Update freq_dict and count_freq for the new count
                freq_dict[id_] = new_count
                count_freq[new_count] += 1
                # Push the new count into the heap
                heapq.heappush(heap, -new_count)
            
            # Determine the current maximum frequency
            current_max = 0
            while heap:
                current_candidate = -heap[0]  # Peek the top element
                if current_candidate in count_freq and count_freq[current_candidate] > 0:
                    current_max = current_candidate
                    break
                else:
                    # Remove the invalid top element
                    heapq.heappop(heap)
            ans.append(current_max)
        
        return ans