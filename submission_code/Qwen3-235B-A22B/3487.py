from typing import List

class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        n = len(source)
        m = len(pattern)
        T = len(targetIndices)
        
        # Create a boolean array to mark which indices are in the targetIndices
        is_target = [False] * n
        for idx in targetIndices:
            is_target[idx] = True
        
        # Initialize the DP array where dp[i] represents the minimal number of target elements used for first i pattern characters
        dp = [float('inf')] * (m + 1)
        dp[0] = 0  # Base case: 0 characters matched uses 0 target elements
        
        # Iterate over each character in the source
        for j in range(n):
            curr_char = source[j]
            # Iterate over pattern indices in reverse order to prevent overwriting needed values
            # We go up to min(m, j+1) to avoid exceeding the pattern length and source index range
            for i in range(min(m, j + 1), 0, -1):
                if pattern[i-1] == curr_char:
                    # If current character matches the pattern's (i-1)-th character, update dp[i]
                    if dp[i-1] != float('inf'):
                        additional_cost = 1 if is_target[j] else 0
                        new_cost = dp[i-1] + additional_cost
                        if new_cost < dp[i]:
                            dp[i] = new_cost
        
        # The minimal number of target indices used in a valid matching of the entire pattern
        minimal_used = dp[m]
        # The maximum number of target indices we can remove is total targets minus those used
        return T - minimal_used