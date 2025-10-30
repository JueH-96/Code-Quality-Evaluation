class Solution:
    def maxScore(self, nums: List[int]) -> int:
        import math
        
        def gcd_of_list(nums):
            if not nums:
                return 0
            result = nums[0]
            for num in nums[1:]:
                result = math.gcd(result, num)
            return result

        def lcm_of_list(nums):
            if not nums:
                return 0
            result = nums[0]
            for num in nums[1:]:
                result = (result * num) // math.gcd(result, num)
            return result
            
        # Calculate the factor score of the original array
        max_score = gcd_of_list(nums) * lcm_of_list(nums)
        
        # Try removing each element and calculate the score
        for i in range(len(nums)):
            new_array = nums[:i] + nums[i+1:]
            new_score = gcd_of_list(new_array) * lcm_of_list(new_array)
            max_score = max(max_score, new_score)
        
        return max_score