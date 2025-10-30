class Solution:
    def maxSum(self, nums: List[int]) -> int:
        unique_nums = list(set(nums))
        max_single = -float('inf')
        total_positive = 0
        has_positive = False
        
        for num in unique_nums:
            if num > 0:
                total_positive += num
                has_positive = True
            else:
                if num > max_single:
                    max_single = num
        
        if has_positive:
            return total_positive
        else:
            return max_single