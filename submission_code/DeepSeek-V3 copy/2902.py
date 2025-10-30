class Solution:
    def maxSum(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        def get_max_digit(num):
            return max(int(d) for d in str(num))
        
        max_digit_to_nums = defaultdict(list)
        
        for num in nums:
            max_digit = get_max_digit(num)
            max_digit_to_nums[max_digit].append(num)
        
        max_sum = -1
        
        for key in max_digit_to_nums:
            if len(max_digit_to_nums[key]) >= 2:
                sorted_nums = sorted(max_digit_to_nums[key], reverse=True)
                current_sum = sorted_nums[0] + sorted_nums[1]
                if current_sum > max_sum:
                    max_sum = current_sum
        
        return max_sum