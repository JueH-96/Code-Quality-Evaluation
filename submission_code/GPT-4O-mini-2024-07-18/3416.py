class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        n = len(nums)
        digit_length = len(str(nums[0]))
        total_difference = 0
        
        for i in range(digit_length):
            count = [0] * 10  # Count of each digit from 0 to 9
            
            for num in nums:
                digit = (num // (10 ** (digit_length - 1 - i))) % 10
                count[digit] += 1
            
            # For each digit, calculate the contribution to the total difference
            for j in range(10):
                if count[j] > 0:
                    total_difference += count[j] * (n - count[j])  # Count pairs with different digits
        
        return total_difference