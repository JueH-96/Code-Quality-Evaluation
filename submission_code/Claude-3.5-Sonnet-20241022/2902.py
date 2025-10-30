class Solution:
    def maxSum(self, nums: List[int]) -> int:
        max_sum = -1
        n = len(nums)
        
        for i in range(n):
            for j in range(i+1, n):
                # Get max digit of both numbers
                max_digit_i = max(str(nums[i]))
                max_digit_j = max(str(nums[j]))
                
                # If max digits are equal, update max_sum if current sum is larger
                if max_digit_i == max_digit_j:
                    curr_sum = nums[i] + nums[j]
                    max_sum = max(max_sum, curr_sum)
                    
        return max_sum