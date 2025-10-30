class Solution:
    def maxScore(self, nums: List[int]) -> int:
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        def lcm(a, b):
            return (a * b) // gcd(a, b)
        
        def array_gcd(arr):
            if not arr:
                return 0
            result = arr[0]
            for i in range(1, len(arr)):
                result = gcd(result, arr[i])
            return result
        
        def array_lcm(arr):
            if not arr:
                return 0
            result = arr[0]
            for i in range(1, len(arr)):
                result = lcm(result, arr[i])
            return result
        
        if len(nums) == 1:
            return nums[0] * nums[0]
        
        max_score = array_gcd(nums) * array_lcm(nums)
        
        # Try removing each element and calculate score
        for i in range(len(nums)):
            remaining = nums[:i] + nums[i+1:]
            curr_gcd = array_gcd(remaining)
            curr_lcm = array_lcm(remaining)
            max_score = max(max_score, curr_gcd * curr_lcm)
            
        return max_score