class Solution:
    def maxLength(self, nums: List[int]) -> int:
        import math
        
        n = len(nums)
        max_length = 0
        
        for i in range(n):
            running_gcd = nums[i]
            running_lcm = nums[i]
            running_product = nums[i]
            
            if running_product == running_gcd * running_lcm:
                max_length = max(max_length, 1)
            
            for j in range(i+1, n):
                running_product *= nums[j]
                running_gcd = math.gcd(running_gcd, nums[j])
                running_lcm = (running_lcm * nums[j]) // math.gcd(running_lcm, nums[j])
                
                if running_product == running_gcd * running_lcm:
                    max_length = max(max_length, j - i + 1)
        
        return max_length