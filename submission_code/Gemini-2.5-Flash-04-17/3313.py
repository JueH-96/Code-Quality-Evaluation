from typing import List
import sys

class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # dp[j][state]
        # j: number of subarrays selected (0 to k)
        # state: 0 if the j-th subarray ends before the current index `i` (i.e., uses elements from nums[0...i-1])
        #        1 if the j-th subarray ends exactly at the current index `i` (i.e., uses element nums[i] as the right endpoint)
        
        # Initialize dp table with negative infinity
        neg_inf = -float('inf')
        
        # prev_dp stores results for nums[0...i-1]
        # curr_dp stores results for nums[0...i]
        # dp table size (k+1) x 2
        prev_dp = [[neg_inf] * 2 for _ in range(k + 1)]
        
        # Base case: Using 0 elements (before processing index 0), 0 subarrays, strength 0
        prev_dp[0][0] = 0
        
        # Iterate through the array elements nums[i] from index 0 to n-1
        for i in range(n):
            # Create the new DP row for current index `i`
            curr_dp = [[neg_inf] * 2 for _ in range(k + 1)]
            
            # Calculate curr_dp[j][1] for j from 1 to k
            # This state means the j-th subarray ends exactly at index i.
            for j in range(1, k + 1):
                # Calculate weight for the j-th subarray (order j)
                # The problem uses 1-based indexing for subarrays i=1...x
                # Strength = sum of (-1)^(i+1) * sum[i] * (x - i + 1)
                # For the j-th subarray (order j), the weight is (-1)^(j+1) * (k - j + 1)
                weight = (k - j + 1) * (1 if (j + 1) % 2 == 0 else -1) # Equivalent to (-1)**(j+1) * (k - j + 1)

                # Case 1: Extend the j-th subarray that ended at index i-1
                # This is possible only if prev_dp[j][1] is not -inf
                extend_strength = neg_inf
                if prev_dp[j][1] != neg_inf:
                    extend_strength = prev_dp[j][1] + nums[i] * weight
                
                # Case 2: Start a new j-th subarray at index i
                # The previous j-1 subarrays must be selected from nums[0...i-1].
                # The max strength for j-1 subarrays from nums[0...i-1] is max(prev_dp[j-1][0], prev_dp[j-1][1]).
                # This is possible only if max(prev_dp[j-1][0], prev_dp[j-1][1]) is not -inf.
                # Note: When j=1, j-1=0. prev_dp[0][0] is 0, prev_dp[0][1] is neg_inf. max(0, neg_inf)=0. This is correct.
                start_strength = neg_inf
                # Ensure index j-1 is valid (always true for j >= 1)
                prev_max_strength_before_i = max(prev_dp[j-1][0], prev_dp[j-1][1])
                
                if prev_max_strength_before_i != neg_inf:
                    start_strength = prev_max_strength_before_i + nums[i] * weight
                    
                curr_dp[j][1] = max(extend_strength, start_strength)

            # Calculate curr_dp[j][0] for j from 0 to k
            # This state means j subarrays are selected from nums[0...i],
            # but the j-th subarray (if j>0) ends at an index strictly less than i.
            # This is equivalent to saying the j subarrays are entirely selected from nums[0...i-1].
            # The max strength is max(prev_dp[j][0], prev_dp[j][1]).
            for j in range(k + 1): # Loop from 0 to k
                 curr_dp[j][0] = max(prev_dp[j][0], prev_dp[j][1])
                 
            # Update prev_dp to curr_dp for the next iteration
            prev_dp = curr_dp

        # After iterating through all elements from 0 to n-1,
        # prev_dp contains the results for nums[0...n-1].
        # The maximum strength using exactly k subarrays from nums[0...n-1]
        # is the maximum of the states where we have selected k subarrays,
        # regardless of whether the k-th subarray ends at index n-1 or before n-1.
        ans = max(prev_dp[k][0], prev_dp[k][1])
        
        return ans