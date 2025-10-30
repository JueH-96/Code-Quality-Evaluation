class Solution:
    def maxLength(self, nums: List[int]) -> int:
        def gcd(a: int, b: int) -> int:
            while b:
                a, b = b, a % b
            return a
        
        def lcm(a: int, b: int) -> int:
            return (a * b) // gcd(a, b)
        
        def array_gcd(arr) -> int:
            result = arr[0]
            for i in range(1, len(arr)):
                result = gcd(result, arr[i])
            return result
        
        def array_lcm(arr) -> int:
            result = arr[0]
            for i in range(1, len(arr)):
                result = lcm(result, arr[i])
            return result
        
        def array_product(arr) -> int:
            result = 1
            for num in arr:
                result *= num
            return result
        
        def is_product_equivalent(arr) -> bool:
            if not arr:
                return False
            prod = array_product(arr)
            gcd_val = array_gcd(arr)
            lcm_val = array_lcm(arr)
            return prod == (lcm_val * gcd_val)
        
        n = len(nums)
        max_length = 1
        
        # Try all possible subarrays
        for i in range(n):
            for j in range(i, n):
                subarray = nums[i:j+1]
                if is_product_equivalent(subarray):
                    max_length = max(max_length, j - i + 1)
        
        return max_length