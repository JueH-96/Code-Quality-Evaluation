from typing import List

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        # Prepare sorted intervals sorted by end time, along with their original indices
        sorted_intervals = []
        for idx, (s, e, w) in enumerate(intervals):
            sorted_intervals.append((s, e, w, idx))
        sorted_intervals.sort(key=lambda x: (x[1], x[0]))  # Sort by end then start
        
        n = len(sorted_intervals)
        ends = [interval[1] for interval in sorted_intervals]
        starts = [interval[0] for interval in sorted_intervals]
        original_indices = [interval[3] for interval in sorted_intervals]
        weights = [interval[2] for interval in sorted_intervals]
        
        # Initialize DP arrays. dp[k] = (max_sum, list of sorted original indices)
        dp = [(0, []) for _ in range(5)]  # k from 0 to 4
        
        for i in range(n):
            s_i = starts[i]
            e_i = ends[i]
            w_i = weights[i]
            orig_i = original_indices[i]
            
            # Binary search to find best_j (largest index j with end < s_i)
            low, high = 0, i - 1
            best_j = -1
            while low <= high:
                mid = (low + high) // 2
                if ends[mid] < s_i:
                    best_j = mid
                    low = mid + 1
                else:
                    high = mid - 1
            
            # Create new_dp to store updated values
            new_dp = [(entry[0], entry[1][:]) for entry in dp]
            
            # Process k from 4 downto 1
            for k in range(4, 0, -1):
                prev_k = k - 1
                # Determine prev_sum and prev_list
                if best_j >= 0:
                    prev_sum, prev_list = dp[prev_k][0], dp[prev_k][1]
                else:
                    if prev_k == 0:
                        prev_sum, prev_list = 0, []
                    else:
                        continue  # No valid previous combination for prev_k
                
                candidate_sum = prev_sum + w_i
                new_list = sorted(prev_list + [orig_i])
                
                # Compare and update if better or lex smaller
                current_sum, current_list = new_dp[k]
                if candidate_sum > current_sum:
                    new_dp[k] = (candidate_sum, new_list)
                elif candidate_sum == current_sum:
                    if new_list < current_list:
                        new_dp[k] = (candidate_sum, new_list)
            
            # Update dp to new_dp
            dp = new_dp
        
        # Find the best combination among k=1 to 4
        best_sum = -1
        best_list = []
        for k in range(1, 5):
            current_sum, current_list = dp[k]
            if current_sum > best_sum:
                best_sum = current_sum
                best_list = current_list
            elif current_sum == best_sum:
                if not best_list:  # if best_list is empty for some reason
                    best_list = current_list
                else:
                    if current_list < best_list:
                        best_list = current_list
        return best_list