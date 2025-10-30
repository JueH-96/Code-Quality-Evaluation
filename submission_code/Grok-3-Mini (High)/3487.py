import bisect
from typing import List

class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        n = len(source)
        len_p = len(pattern)
        target_set = set(targetIndices)
        
        # Initialize dp for the first character of pattern (j=0)
        dp_list = []
        for i in range(n):
            if source[i] == pattern[0]:
                cost_i = 1 if i in target_set else 0
                dp_list.append((i, cost_i))
        
        # Compute dp for each subsequent character in pattern
        for j in range(1, len_p):
            prev_dp_list = dp_list
            if not prev_dp_list:
                # No valid endings for previous pattern part, should not happen but handle
                dp_list = []
                continue
            
            prev_k = [pair[0] for pair in prev_dp_list]
            prev_dp_values = [pair[1] for pair in prev_dp_list]
            
            # Compute cumulative minimum of previous dp values
            cum_min_list = [prev_dp_values[0]]
            for val in prev_dp_values[1:]:
                cum_min_list.append(min(cum_min_list[-1], val))
            
            # Compute new dp list for current pattern character
            new_dp_list = []
            for i in range(n):
                if source[i] == pattern[j]:
                    # Find the minimum cost ending before index i
                    num_less = bisect.bisect_left(prev_k, i)
                    if num_less > 0:
                        min_over_k = cum_min_list[num_less - 1]
                    else:
                        min_over_k = float('inf')
                    
                    if min_over_k < float('inf'):
                        cost_i = 1 if i in target_set else 0
                        dp_cost = cost_i + min_over_k
                        new_dp_list.append((i, dp_cost))
            
            # Update dp_list for the next iteration
            dp_list = new_dp_list
        
        # Find the minimum cost to match the entire pattern
        if not dp_list:
            # This should not happen as pattern is guaranteed to be a subsequence
            return 0  # Fallback, though input guarantees a match exists
        
        min_cost = min(pair[1] for pair in dp_list)
        
        # Maximum removals is the size of targetIndices minus the minimum cost
        return len(targetIndices) - min_cost