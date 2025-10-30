class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        # Sort intervals by their right endpoint, then by original index to maintain lex order
        intervals_sorted = sorted(enumerate(intervals), key=lambda x: (x[1][1], x[0]))
        
        # Initialize DP: each entry tracks sum, end position, and indices
        dp = [
            {'sum': 0, 'end': float('-inf'), 'indices': []},  # 0 intervals
            {'sum': float('-inf'), 'end': None, 'indices': None},  # 1 interval
            {'sum': float('-inf'), 'end': None, 'indices': None},  # 2 intervals
            {'sum': float('-inf'), 'end': None, 'indices': None},  # 3 intervals
            {'sum': float('-inf'), 'end': None, 'indices': None},  # 4 intervals
        ]
        
        for interval_info in intervals_sorted:
            original_index, interval = interval_info
            l, r, w = interval  # l: left, r: right, w: weight
            
            # Check each possible number of intervals from 4 down to 1
            for m in range(4, 0, -1):
                prev_m = m - 1
                prev_dp = dp[prev_m]
                
                # If previous state is invalid, skip
                if prev_dp['sum'] == float('-inf'):
                    continue
                
                # Check if current interval starts after the previous end
                if l <= prev_dp['end']:
                    continue  # Overlapping, skip
                
                # Calculate new potential state
                new_sum = prev_dp['sum'] + w
                new_end = r
                new_indices = prev_dp['indices'].copy()
                new_indices.append(original_index)
                new_indices.sort()  # Maintain lex order
                
                current_dp = dp[m]
                # Update current_dp if new_sum is better or lex smaller
                if new_sum > current_dp['sum']:
                    current_dp['sum'] = new_sum
                    current_dp['end'] = new_end
                    current_dp['indices'] = new_indices
                elif new_sum == current_dp['sum']:
                    if current_dp['indices'] is None or new_indices < current_dp['indices']:
                        current_dp['sum'] = new_sum
                        current_dp['end'] = new_end
                        current_dp['indices'] = new_indices
        
        # Determine the maximum sum and collect all candidates
        max_sum = -float('inf')
        for m in range(1, 5):
            if dp[m]['sum'] > max_sum:
                max_sum = dp[m]['sum']
        
        candidates = []
        for m in range(1, 5):
            if dp[m]['sum'] == max_sum:
                candidates.append(dp[m]['indices'])
        
        # Return the lexicographically smallest candidate
        return min(candidates)