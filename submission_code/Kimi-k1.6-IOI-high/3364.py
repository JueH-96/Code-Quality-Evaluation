class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        m = len(andValues)
        n = len(nums)
        
        # Edge case where m exceeds n, impossible to split
        if m > n:
            return -1
        
        INF = float('inf')
        dp = [[INF] * (n + 1) for _ in range(m + 1)]
        dp[0][0] = 0  # Base case: 0 subarrays for 0 elements
        
        for k in range(1, m + 1):
            required_and = andValues[k - 1]
            prev_dict = {}
            
            for i in range(1, n + 1):
                current_element = nums[i - 1]
                current_dict = {}
                
                # Case 1: Subarray starts at current element (i-1)
                if dp[k - 1][i - 1] != INF:
                    and_val = current_element
                    if and_val in current_dict:
                        current_dict[and_val] = min(current_dict[and_val], dp[k - 1][i - 1])
                    else:
                        current_dict[and_val] = dp[k - 1][i - 1]
                
                # Case 2: Extend subarrays from previous positions
                for and_val in prev_dict:
                    new_and = and_val & current_element
                    if new_and in current_dict:
                        current_dict[new_and] = min(current_dict[new_and], prev_dict[and_val])
                    else:
                        current_dict[new_and] = prev_dict[and_val]
                
                # Update DP if required AND is found
                if required_and in current_dict:
                    candidate = current_dict[required_and] + current_element
                    if candidate < dp[k][i]:
                        dp[k][i] = candidate
                
                # Update prev_dict for next iteration
                prev_dict = current_dict.copy()  # Shallow copy is sufficient here
        
        return dp[m][n] if dp[m][n] != INF else -1