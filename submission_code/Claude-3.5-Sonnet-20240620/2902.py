class Solution:
    def maxSum(self, nums: List[int]) -> int:
        def max_digit(num):
            return max(int(digit) for digit in str(num))
        
        max_sum = -1
        digit_dict = {}
        
        for num in nums:
            max_d = max_digit(num)
            if max_d in digit_dict:
                max_sum = max(max_sum, num + digit_dict[max_d])
                digit_dict[max_d] = max(digit_dict[max_d], num)
            else:
                digit_dict[max_d] = num
        
        return max_sum