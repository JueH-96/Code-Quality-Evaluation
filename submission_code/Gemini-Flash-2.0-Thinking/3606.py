class Solution:
    def minElement(self, nums: List[int]) -> int:
        modified_nums = []
        for num in nums:
            digit_sum = 0
            for digit in str(num):
                digit_sum += int(digit)
            modified_nums.append(digit_sum)
        
        return min(modified_nums)