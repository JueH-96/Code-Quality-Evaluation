class Solution:
    def maxLength(self, nums: List[int]) -> int:
        import math
        
        def gcd_of_array(arr):
            result = arr[0]
            for x in arr[1:]:
                result = math.gcd(result, x)
            return result
        
        def lcm_of_array(arr):
            result = arr[0]
            for x in arr[1:]:
                result = result * x // math.gcd(result, x)
            return result
        
        def product_of_array(arr):
            result = 1
            for x in arr:
                result *= x
            return result
        
        n = len(nums)
        max_length = 0
        
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                prod = product_of_array(subarray)
                gcd_val = gcd_of_array(subarray)
                lcm_val = lcm_of_array(subarray)
                
                if prod == lcm_val * gcd_val:
                    max_length = max(max_length, len(subarray))
        
        return max_length