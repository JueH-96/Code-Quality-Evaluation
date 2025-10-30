import heapq
from collections import defaultdict

class Solution:
    def mostFrequentIDs(self, nums: List[int], freq: List[int]) -> List[int]:
        counts = defaultdict(int)
        freq_map = defaultdict(int)
        heap = []
        ans = []
        for i in range(len(nums)):
            current_id = nums[i]
            current_freq = freq[i]
            previous_count = counts[current_id]
            new_count = previous_count + current_freq
            counts[current_id] = new_count
            heapq.heappush(heap, -new_count)
            # Update the frequency map
            if previous_count != 0:
                freq_map[previous_count] -= 1
                if freq_map[previous_count] == 0:
                    del freq_map[previous_count]
            freq_map[new_count] += 1
            # Find the current maximum frequency
            current_max = 0
            while heap:
                current_candidate = -heap[0]
                if freq_map.get(current_candidate, 0) > 0:
                    current_max = current_candidate
                    break
                else:
                    heapq.heappop(heap)
            ans.append(current_max)
        return ans