import bisect

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        # Sort intervals by their end time and keep track of original indices
        sorted_intervals = sorted(
            [(l, r, w, idx) for idx, (l, r, w) in enumerate(intervals)],
            key=lambda x: x[1]
        )
        
        # Initialize DP arrays. Each dp[k] is a list of tuples (end_time, weight, indices)
        dp = [[] for _ in range(5)]
        dp[0].append((-float('inf'), 0, []))  # Initial state for 0 intervals
        
        for l, r, w, idx in sorted_intervals:
            # Iterate backwards to prevent overwriting the previous states
            for k in range(4, 0, -1):
                prev_list = dp[k-1]
                if not prev_list:
                    continue
                
                # Binary search to find the best previous state with end_time <= l
                low, high = 0, len(prev_list) - 1
                best_idx = -1
                while low <= high:
                    mid = (low + high) // 2
                    if prev_list[mid][0] <= l:
                        best_idx = mid
                        low = mid + 1
                    else:
                        high = mid - 1
                
                if best_idx != -1:
                    prev_e, prev_w, prev_indices = prev_list[best_idx]
                    new_e = r
                    new_w = prev_w + w
                    new_indices = prev_indices + [idx]
                    
                    candidate = (new_e, new_w, new_indices)
                    current_list = dp[k]
                    
                    # Find the position to insert the candidate
                    pos = bisect.bisect_right([x[0] for x in current_list], new_e)
                    temp_list = current_list[:pos] + [candidate] + current_list[pos:]
                    
                    # Prune the list to maintain the properties: end times are increasing and weights are non-decreasing
                    pruned = []
                    for entry in reversed(temp_list):
                        # Check if the current entry is not dominated by the last entry in pruned
                        while pruned:
                            last = pruned[-1]
                            if (entry[0] <= last[0] and entry[1] <= last[1]):
                                # entry is dominated by last entry
                                break
                            elif (entry[0] == last[0] and entry[1] == last[1]):
                                # same end and weight, choose lex smaller indices
                                if entry[2] < last[2]:
                                    pruned.pop()
                                else:
                                    break
                            else:
                                break
                        else:
                            pruned.append(entry)
                    pruned.reverse()
                    dp[k] = pruned
        
        # Find the best indices among all possible counts (1 to 4)
        max_weight = -1
        best_indices = []
        for k in range(4, 0, -1):
            if dp[k]:
                current_max = -1
                current_indices = []
                for entry in dp[k]:
                    if entry[1] > current_max or (entry[1] == current_max and entry[2] < current_indices):
                        current_max = entry[1]
                        current_indices = entry[2]
                if current_max > max_weight or (current_max == max_weight and current_indices < best_indices):
                    max_weight = current_max
                    best_indices = current_indices
        
        return best_indices