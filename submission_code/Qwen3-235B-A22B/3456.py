from typing import List

class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        # Initialize dp: list of dictionaries. Each dp[d] is a dictionary of last element to max length.
        dp = [{} for _ in range(k + 1)]
        
        for x in nums:
            # Create new_dp as a copy of current dp
            new_dp = [{} for _ in range(k + 1)]
            for d in range(k + 1):
                # Copy current dp[d] into new_dp[d]
                for key in dp[d]:
                    new_dp[d][key] = dp[d][key]
            
            # Process all previous states to update new_dp
            for d in range(k + 1):
                for key in dp[d]:
                    current_length = dp[d][key]
                    if x == key:
                        # Append x to this subsequence, transitions remain d
                        if new_dp[d].get(key, 0) < current_length + 1:
                            new_dp[d][key] = current_length + 1
                    else:
                        new_d = d + 1
                        if new_d <= k:
                            # Append x to this subsequence, transitions increase by 1
                            if new_dp[new_d].get(x, 0) < current_length + 1:
                                new_dp[new_d][x] = current_length + 1
            
            # Add x as a new subsequence of length 1
            if new_dp[0].get(x, 0) < 1:
                new_dp[0][x] = 1
            
            # Update dp to be new_dp
            dp = new_dp
        
        # Find the maximum length across all dp entries
        res = 0
        for d in range(k + 1):
            if dp[d]:
                current_max = max(dp[d].values())
                res = max(res, current_max)
        return res