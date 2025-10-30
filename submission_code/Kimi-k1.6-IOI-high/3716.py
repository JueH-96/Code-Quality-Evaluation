class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        # Initialize max_suffix with -infinity for all values and differences
        max_suffix = [[-float('inf')] * 300 for _ in range(301)]  # values 0-300 (0 unused)
        max_length = 1
        
        for x in nums:
            dp = [1] * 300
            for d_new in range(300):
                v1 = x - d_new
                v2 = x + d_new
                candidates = []
                # Check if v1 is a valid value (1 <= v1 <= 300)
                if 1 <= v1 <= 300:
                    if max_suffix[v1][d_new] != -float('inf'):
                        candidates.append(max_suffix[v1][d_new])
                # Check if v2 is a valid value (1 <= v2 <= 300)
                if 1 <= v2 <= 300:
                    if max_suffix[v2][d_new] != -float('inf'):
                        candidates.append(max_suffix[v2][d_new])
                # Update dp[d_new] if there are valid candidates
                if candidates:
                    current_max = max(candidates)
                    if dp[d_new] < current_max + 1:
                        dp[d_new] = current_max + 1
            
            # Compute suffix_max for the current dp array
            suffix_max = [0] * 300
            suffix_max[299] = dp[299]
            for d in range(298, -1, -1):
                suffix_max[d] = max(dp[d], suffix_max[d + 1])
            
            # Update the max_suffix for the current value x
            for d in range(300):
                if suffix_max[d] > max_suffix[x][d]:
                    max_suffix[x][d] = suffix_max[d]
            
            # Update the global maximum length
            current_max = max(dp)
            if current_max > max_length:
                max_length = current_max
        
        return max_length