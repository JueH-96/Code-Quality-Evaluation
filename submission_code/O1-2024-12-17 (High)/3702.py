class Solution:
    def maxLength(self, nums: List[int]) -> int:
        from math import gcd
        
        n = len(nums)
        # We'll keep track of the maximum length found.
        max_len = 1   # Subarrays of length 1 can be valid if the element is 1.
        
        # Check all subarrays [i..j].
        for i in range(n):
            gcd_val = 0
            lcm_val = 1
            product_val = 1
            for j in range(i, n):
                # Update gcd, product, and lcm for the new element nums[j].
                gcd_val = gcd(gcd_val, nums[j])
                product_val *= nums[j]
                
                # Compute lcm step by step using the gcd.
                g = gcd(lcm_val, nums[j])
                lcm_val = (lcm_val * nums[j]) // g
                
                # Check if the product-equivalent condition holds.
                if gcd_val * lcm_val == product_val:
                    max_len = max(max_len, j - i + 1)
        
        return max_len