import bisect
from typing import List

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        # Sort intervals by their end time and keep track of original indices
        sorted_intervals = sorted(
            [(interval[0], interval[1], interval[2], idx) for idx, interval in enumerate(intervals)],
            key=lambda x: x[1]
        )
        
        # Initialize entries for each k (1-4). Each entry is a tuple (end_time, sum, indices)
        k_entries = [[] for _ in range(5)]  # k_entries[0] is unused
        
        for interval in sorted_intervals:
            start_i, end_i, weight_i, original_idx = interval
            
            # Process each possible count of intervals from 4 down to 1
            for k in range(4, 0, -1):
                if k == 1:
                    new_sum = weight_i
                    new_indices = [original_idx]
                    new_end = end_i
                    
                    # Check if this is better than existing entries in k=1's list
                    if not k_entries[k]:
                        k_entries[k].append((new_end, new_sum, new_indices))
                    else:
                        last_end, last_sum, last_indices = k_entries[k][-1]
                        if new_sum > last_sum:
                            k_entries[k].append((new_end, new_sum, new_indices))
                        elif new_sum == last_sum:
                            if new_indices < last_indices:
                                k_entries[k].append((new_end, new_sum, new_indices))
                else:
                    # Check if there are entries in k-1's list
                    if not k_entries[k-1]:
                        continue
                    
                    # Binary search to find the latest end_time <= start_i in k-1's entries
                    end_times = [entry[0] for entry in k_entries[k-1]]
                    idx = bisect.bisect_right(end_times, start_i) - 1
                    if idx < 0:
                        continue  # No valid previous interval found
                    
                    # Get the best previous entry for k-1
                    prev_entry = k_entries[k-1][idx]
                    prev_sum = prev_entry[1]
                    prev_indices = prev_entry[2]
                    new_sum = prev_sum + weight_i
                    new_indices = sorted(prev_indices + [original_idx])
                    new_end = end_i
                    
                    # Check if this new entry is better than existing entries in k's list
                    if not k_entries[k]:
                        k_entries[k].append((new_end, new_sum, new_indices))
                    else:
                        last_end, last_sum, last_indices = k_entries[k][-1]
                        if new_sum > last_sum:
                            k_entries[k].append((new_end, new_sum, new_indices))
                        elif new_sum == last_sum:
                            if new_indices < last_indices:
                                k_entries[k].append((new_end, new_sum, new_indices))
        
        # Determine the maximum sum across all k's
        max_sum = 0
        for k in range(1, 5):
            if k_entries[k]:
                current_max = k_entries[k][-1][1]
                if current_max > max_sum:
                    max_sum = current_max
        
        # Collect all candidates with sum equal to max_sum
        candidates = []
        for k in range(1, 5):
            if not k_entries[k]:
                continue
            # Find all entries in k_entries[k] with sum == max_sum
            sum_list = k_entries[k]
            start = len(sum_list) - 1
            while start >= 0 and sum_list[start][1] == max_sum:
                start -= 1
            # Collect entries from start+1 to end
            for entry in sum_list[start+1:]:
                candidates.append(entry[2])
        
        # Return the lexicographically smallest candidate if any
        return min(candidates) if candidates else []