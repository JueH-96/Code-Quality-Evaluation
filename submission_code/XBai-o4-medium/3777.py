from typing import List

class Solution:
    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
        sum_nums = sum(nums)
        if k < -sum_nums or k > sum_nums:
            return -1
        
        # Initialize DP: dp[0] for even parity (even length), dp[1] for odd
        dp = [{0: 1}, {}]  # dp[0] is a dictionary, dp[1] is a dictionary
        
        for num in nums:
            # Create a temporary copy of the current dp
            temp_dp = [dict(dp[0]), dict(dp[1])]
            
            # Process adding the current num to existing states
            for parity in [0, 1]:
                for s in list(dp[parity].keys()):
                    current_product = dp[parity][s]
                    new_parity = 1 - parity
                    delta = num if parity == 0 else -num
                    new_sum = s + delta
                    new_product = current_product * num
                    
                    if new_product > limit:
                        continue
                    
                    # Update the temp_dp
                    if new_sum in temp_dp[new_parity]:
                        if temp_dp[new_parity][new_sum] < new_product:
                            temp_dp[new_parity][new_sum] = new_product
                    else:
                        temp_dp[new_parity][new_sum] = new_product
            
            # Update dp to temp_dp
            dp[0] = temp_dp[0]
            dp[1] = temp_dp[1]
        
        max_prod = -1
        for parity in [0, 1]:
            if k in dp[parity]:
                product = dp[parity][k]
                # Check if it's the empty subsequence
                if product == 1 and k == 0 and parity == 0:
                    continue
                if product > max_prod:
                    max_prod = product
        
        return max_prod if max_prod != -1 else -1