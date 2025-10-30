class Solution:
    def maxLength(self, nums: List[int]) -> int:
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        def lcm(a, b):
            return a * b // gcd(a, b)
        
        n = len(nums)
        max_len = 0
        
        for i in range(n):
            prod = nums[i]
            arr_gcd = nums[i]
            arr_lcm = nums[i]
            
            # Check single element subarray
            if prod == arr_lcm * arr_gcd:
                max_len = max(max_len, 1)
            
            # Extend subarray to the right
            for j in range(i + 1, n):
                prod *= nums[j]
                arr_gcd = gcd(arr_gcd, nums[j])
                arr_lcm = lcm(arr_lcm, nums[j])
                
                if prod == arr_lcm * arr_gcd:
                    max_len = max(max_len, j - i + 1)
        
        return max_len