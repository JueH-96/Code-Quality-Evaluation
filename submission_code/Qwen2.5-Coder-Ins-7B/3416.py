class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        total_sum = 0
        num_strs = [str(num) for num in nums]
        num_length = len(num_strs[0])
        
        for i in range(num_length):
            digit_count = [0] * 10
            for num_str in num_strs:
                digit_count[int(num_str[i])] += 1
            
            for j in range(10):
                total_sum += (digit_count[j] * (len(nums) - digit_count[j]) * (num_length - i))
        
        return total_sum