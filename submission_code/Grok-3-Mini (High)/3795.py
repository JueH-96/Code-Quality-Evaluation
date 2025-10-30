from typing import List

class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        q_len = len(queries)
        max_sum = 1000  # Maximum possible sum we need to handle
        min_ks = []
        
        for i in range(n):  # For each index in nums
            target = nums[i]
            dp = [False] * (max_sum + 1)
            dp[0] = True
            
            if dp[target]:  # If target is 0, it's already achievable with no queries
                min_k_i = 0
            else:
                min_k_i = -1  # Initialize to impossible
                for j in range(q_len):  # Iterate through each query
                    l, r, val = queries[j]
                    if l <= i <= r:  # If the query covers index i
                        v = val
                        # Update dp with the new value v
                        for s in range(max_sum, v - 1, -1):  # From max_sum down to v inclusive
                            if dp[s - v]:
                                dp[s] = True  # Set dp[s] to True if dp[s - v] is True
                        # After updating dp, check if target is now achievable
                        if dp[target]:
                            min_k_i = j + 1  # k is j + 1 since queries are 0-based index
                            break  # No need to check further queries for this index
            
            min_ks.append(min_k_i)  # Append the min k for this index, or -1 if impossible
        
        # Now, check if any index cannot be made zero
        if -1 in min_ks:
            return -1
        else:
            # Return the maximum of the min k values
            return max(min_ks)