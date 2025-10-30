class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: list[int]) -> int:
        n = len(source)
        m = len(pattern)
        
        if m == 0:
            return len(targetIndices)

        target_set = set(targetIndices)
        
        # dp_prev corresponds to the DP row for pattern[:i-1].
        # We initialize it for the base case of an empty pattern prefix (i=0),
        # where the cost is 0 for any source prefix length.
        dp_prev = [0] * (n + 1)
        
        # Iterate through each character of the pattern (from 1 to m)
        for i in range(1, m + 1):
            # dp_curr will be the DP row for pattern[:i].
            # Initialize with infinity, as we might not be able to form the subsequence.
            # The 0-th element dp_curr[0] represents matching pattern[:i] in an empty
            # source prefix, which is impossible, so it remains infinity.
            dp_curr = [float('inf')] * (n + 1)
            
            # Iterate through each character of the source (from 1 to n)
            for j in range(1, n + 1):
                # Option 1: Don't use source[j-1] for the match.
                # The cost is the same as matching pattern[:i] in source[:j-1].
                # This value comes from the same row (dp_curr) at the previous column.
                option1_cost = dp_curr[j - 1]
                
                # Option 2: Use source[j-1] to match pattern[i-1], if they are equal.
                option2_cost = float('inf')
                if source[j-1] == pattern[i-1]:
                    # Cost of using source[j-1] is 1 if its index is in target_set, else 0.
                    cost = 1 if (j - 1) in target_set else 0
                    
                    # The total cost is the cost to match pattern[:i-1] in source[:j-1]
                    # (from the previous row dp_prev) plus the cost of using the current character.
                    option2_cost = dp_prev[j - 1] + cost
                
                # The cost for dp[i][j] is the minimum of the two options.
                dp_curr[j] = min(option1_cost, option2_cost)
            
            # The current row becomes the previous row for the next pattern character.
            dp_prev = dp_curr
            
        # After the loops, dp_prev[n] holds dp[m][n], which is the minimum number of 
        # characters from targetIndices that must be kept to form the pattern subsequence.
        min_kept_in_target = dp_prev[n]
        
        # The maximum number of characters we can remove is the total number of
        # target characters minus the minimum we must keep.
        return len(targetIndices) - min_kept_in_target