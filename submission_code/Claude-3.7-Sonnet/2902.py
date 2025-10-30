class Solution:
    def maxSum(self, nums: List[int]) -> int:
        # Helper function to find the maximum digit in a number
        def max_digit(num):
            return max(int(digit) for digit in str(num))
        
        # Group numbers by their maximum digit
        digit_to_nums = {}
        for num in nums:
            digit = max_digit(num)
            if digit in digit_to_nums:
                digit_to_nums[digit].append(num)
            else:
                digit_to_nums[digit] = [num]
        
        # Find the maximum sum for each group with at least 2 numbers
        max_sum = -1
        for nums_list in digit_to_nums.values():
            if len(nums_list) >= 2:
                nums_list.sort(reverse=True)
                max_sum = max(max_sum, nums_list[0] + nums_list[1])
        
        return max_sum