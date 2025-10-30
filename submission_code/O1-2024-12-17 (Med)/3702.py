class Solution:
    def maxLength(self, nums: List[int]) -> int:
        from math import gcd
        
        n = len(nums)
        max_len = 0
        
        # Check all subarrays in O(n^2) fashion
        for start in range(n):
            current_gcd = 0
            current_lcm = 1
            current_prod = 1
            for end in range(start, n):
                x = nums[end]
                # Update the running product
                current_prod *= x
                
                if end == start:
                    # For the first element in this subarray
                    current_gcd = x
                    current_lcm = x
                else:
                    # Update gcd
                    current_gcd = gcd(current_gcd, x)
                    # Update lcm using gcd
                    current_lcm = (current_lcm * x) // gcd(current_lcm, x)
                
                # Check the product-equivalent condition
                if current_prod == current_gcd * current_lcm:
                    max_len = max(max_len, end - start + 1)
        
        return max_len