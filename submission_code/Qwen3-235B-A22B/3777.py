from typing import List

class Solution:
    def maxProduct(self, nums: List[int], k: int, limit: int) -> int:
        # Initialize DP with (parity 0, sum 0) -> product 1 (empty subsequence)
        dp = {(0, 0): 1}
        
        for num in nums:
            new_dp = dict(dp)  # Start with existing states without taking 'num'
            # Iterate through each existing state and generate new states by taking 'num'
            for (p, s), product in dp.items():
                # Calculate new sum and parity after taking 'num'
                new_s = s + (num if p == 0 else -num)
                new_p = (p + 1) % 2
                new_product = product * num
                
                if new_product > limit:
                    continue  # Skip if product exceeds the limit
                
                key = (new_p, new_s)
                # Update new_dp if this product is better than existing entry
                if key in new_dp:
                    if new_product > new_dp[key]:
                        new_dp[key] = new_product
                else:
                    new_dp[key] = new_product
            # Update dp to be new_dp after processing current num
            dp = new_dp
        
        max_prod = -float('inf')
        # Check all states except the empty subsequence
        for (p, s), product in dp.items():
            if (p == 0 and s == 0):
                continue  # Skip empty subsequence
            if s == k:
                if product > max_prod:
                    max_prod = product
        
        return max_prod if max_prod != -float('inf') else -1