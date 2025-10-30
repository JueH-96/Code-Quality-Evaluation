class Solution:
    def maxSum(self, nums: List[int]) -> int:
        def max_digit(num):
            return max(int(digit) for digit in str(num))
        
        max_digit_to_nums = {}
        for num in nums:
            max_dig = max_digit(num)
            if max_dig in max_digit_to_nums:
                max_digit_to_nums[max_dig].append(num)
            else:
                max_digit_to_nums[max_dig] = [num]
        
        max_pair_sum = -1
        for num_list in max_digit_to_nums.values():
            if len(num_list) > 1:
                num_list.sort(reverse=True)
                max_pair_sum = max(max_pair_sum, num_list[0] + num_list[1])
        
        return max_pair_sum