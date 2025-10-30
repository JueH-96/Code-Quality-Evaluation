from typing import List

class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        # Convert each number to its string form.
        # Since all the integers in nums have the same number of digits, 
        # the string representations will also have equal lengths.
        s_nums = [str(num) for num in nums]
        d = len(s_nums[0])
        n = len(nums)
        total_diff = 0
        
        # Process each digit position separately.
        for pos in range(d):
            # Count how many times each digit (0 to 9) appears at this position.
            freq = [0] * 10
            for sn in s_nums:
                digit = int(sn[pos])
                freq[digit] += 1
            
            # For current digit position, count pairs with different digits.
            # For a digit group with frequency f, the number of pairs 
            # where one number shows the digit and the other shows a different digit is f * (n - f).
            # Since each pair will be counted twice (once per each digit), we divide by 2.
            pos_diff = sum(f * (n - f) for f in freq) // 2
            total_diff += pos_diff
        
        return total_diff