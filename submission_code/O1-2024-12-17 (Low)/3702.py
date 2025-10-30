class Solution:
    def maxLength(self, nums: List[int]) -> int:
        import math

        n = len(nums)
        max_len = 1

        for start in range(n):
            gcd_val = nums[start]
            lcm_val = nums[start]
            prod_val = nums[start]
            for end in range(start + 1, n):
                # Update gcd, lcm, product with the new element.
                gcd_val = math.gcd(gcd_val, nums[end])
                # To avoid repeated factor calculation for lcm update:
                lcm_val = (lcm_val * nums[end]) // math.gcd(lcm_val, nums[end])
                prod_val *= nums[end]
                
                # Check if the subarray satisfies prod(arr) == gcd(arr) * lcm(arr)
                if prod_val == gcd_val * lcm_val:
                    max_len = max(max_len, end - start + 1)

            # Single element subarray check (though often trivially product equivalent)
            # But in case no longer subarray found from start, consider single element.
            max_len = max(max_len, 1)

        return max_len