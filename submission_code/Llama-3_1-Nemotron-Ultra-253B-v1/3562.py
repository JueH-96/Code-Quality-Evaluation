from typing import List
import bisect

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        # Sort intervals by end time, then by original index to handle ties
        sorted_intervals = sorted([(l, r, w, i) for i, (l, r, w) in enumerate(intervals)], key=lambda x: (x[1], x[3]))
        
        # Initialize DP states for 0 to 4 intervals
        states = [[] for _ in range(5)]
        states[0] = [(-10**18, 0, [])]  # (end_time, sum, indices)
        
        for l_i, r_i, w_i, original_idx in sorted_intervals:
            # Process k from 3 down to 0 to avoid overwriting the same k in one iteration
            for k in range(3, -1, -1):
                if k + 1 > 4:
                    continue
                current_k_states = states[k]
                if not current_k_states:
                    continue
                
                # Binary search for the latest non-overlapping interval
                end_times = [s[0] for s in current_k_states]
                pos = bisect.bisect_right(end_times, l_i - 1) - 1
                if pos < 0:
                    continue
                
                best_state = current_k_states[pos]
                new_sum = best_state[1] + w_i
                new_indices = best_state[2] + [original_idx]
                new_end = r_i
                new_k = k + 1
                
                target_states = states[new_k]
                target_end_times = [s[0] for s in target_states]
                insert_pos = bisect.bisect_left(target_end_times, new_end)
                
                # Check if new state is dominated by existing states
                prefix_max = []
                current_max = -10**18
                for s in target_states:
                    current_max = max(current_max, s[1])
                    prefix_max.append(current_max)
                if insert_pos > 0 and prefix_max[insert_pos - 1] >= new_sum:
                    continue
                
                # Insert new_state and remove dominated states
                new_state = (new_end, new_sum, new_indices)
                new_target_states = target_states[:insert_pos] + [new_state]
                for s in target_states[insert_pos:]:
                    if s[1] > new_sum:
                        new_target_states.append(s)
                
                # Ensure no duplicates and maintain lex order
                if insert_pos > 0:
                    prev_s = new_target_states[insert_pos - 1]
                    if prev_s[0] == new_end and prev_s[1] == new_sum:
                        if prev_s[2] < new_state[2]:
                            new_target_states.pop(insert_pos)
                        else:
                            new_target_states.pop(insert_pos - 1)
                
                states[new_k] = new_target_states
        
        # Find the maximum sum and corresponding lex smallest indices
        max_sum = -1
        best_indices = []
        for k in range(1, 5):
            if not states[k]:
                continue
            k_max = max(s[1] for s in states[k])
            if k_max < max_sum:
                continue
            candidates = [s for s in states[k] if s[1] == k_max]
            if not candidates:
                continue
            min_candidate = min(candidates, key=lambda x: x[2])
            if k_max > max_sum or (k_max == max_sum and min_candidate[2] < best_indices):
                max_sum = k_max
                best_indices = min_candidate[2]
        
        return best_indices