def is_lex_smaller(a, b):
    min_len = min(len(a), len(b))
    for i in range(min_len):
        if a[i] < b[i]:
            return True
        elif a[i] > b[i]:
            return False
    if len(a) < len(b):
        return True
    else:
        return False

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        # Add original indices
        indexed_intervals = list(enumerate(intervals))
        # Sort intervals by their end times
        sorted_intervals = sorted(indexed_intervals, key=lambda x: x[1][1])
        
        # Initialize DP
        dp = [{} for _ in range(5)]
        initial_last_end = -1e18
        dp[0][initial_last_end] = (0, [])
        
        for original_index, interval in sorted_intervals:
            l = interval[0]
            r = interval[1]
            w = interval[2]
            
            # Create a temporary DP to hold new states
            temp_dp = [{} for _ in range(5)]
            
            # Copy the current DP into the temp_dp
            for k in range(5):
                for last_end in dp[k]:
                    sum_val, indices = dp[k][last_end]
                    # Add this state to the temp_dp
                    if last_end in temp_dp[k]:
                        existing_sum, existing_indices = temp_dp[k][last_end]
                        if sum_val > existing_sum:
                            temp_dp[k][last_end] = (sum_val, indices.copy())
                        elif sum_val == existing_sum:
                            if is_lex_smaller(indices, existing_indices):
                                temp_dp[k][last_end] = (sum_val, indices.copy())
                    else:
                        temp_dp[k][last_end] = (sum_val, indices.copy())
            
            # Process the current interval
            for k in range(4, -1, -1):
                for last_end in dp[k]:
                    sum_val, indices = dp[k][last_end]
                    if l > last_end:
                        new_k = k + 1
                        if new_k > 4:
                            continue
                        new_last_end = r
                        new_sum = sum_val + w
                        new_indices = indices + [original_index]
                        new_indices.sort()  # Maintain sorted order
                        
                        # Add this new state to the temp_dp
                        if new_last_end in temp_dp[new_k]:
                            existing_sum, existing_indices = temp_dp[new_k][new_last_end]
                            if new_sum > existing_sum:
                                temp_dp[new_k][new_last_end] = (new_sum, new_indices)
                            elif new_sum == existing_sum:
                                if is_lex_smaller(new_indices, existing_indices):
                                    temp_dp[new_k][new_last_end] = (new_sum, new_indices)
                        else:
                            temp_dp[new_k][new_last_end] = (new_sum, new_indices)
            
            # Update the DP to the temp_dp
            dp = temp_dp
        
        # Find the best state
        max_sum = -1
        best_indices = None
        for k in range(5):
            for last_end in dp[k]:
                sum_val, indices = dp[k][last_end]
                if sum_val > max_sum:
                    max_sum = sum_val
                    best_indices = indices
                elif sum_val == max_sum:
                    if best_indices is None or is_lex_smaller(indices, best_indices):
                        best_indices = indices
        
        if best_indices is None:
            return []
        else:
            return best_indices