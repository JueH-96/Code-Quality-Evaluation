class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        if not nums:
            return 0
        
        # Convert all numbers to strings to easily access digits
        str_nums = [str(num) for num in nums]
        num_digits = len(str_nums[0])
        n = len(nums)
        
        total_diff = 0
        
        # For each digit position
        for pos in range(num_digits):
            # Count frequency of each digit at this position
            digit_count = {}
            for num_str in str_nums:
                digit = num_str[pos]
                digit_count[digit] = digit_count.get(digit, 0) + 1
            
            # For this position, calculate how many pairs have different digits
            # For each pair of different digits, multiply their counts
            digits = list(digit_count.keys())
            for i in range(len(digits)):
                for j in range(i + 1, len(digits)):
                    # All pairs between digit i and digit j at this position
                    total_diff += digit_count[digits[i]] * digit_count[digits[j]]
        
        return total_diff