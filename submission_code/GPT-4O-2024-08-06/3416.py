class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        # Convert all numbers to strings for easy digit comparison
        str_nums = list(map(str, nums))
        
        # Get the number of digits in each number (all have the same number of digits)
        num_digits = len(str_nums[0])
        
        # Initialize the total sum of digit differences
        total_diff = 0
        
        # Iterate over each digit position
        for i in range(num_digits):
            # Count the frequency of each digit at position i
            digit_count = [0] * 10
            for num in str_nums:
                digit = int(num[i])
                digit_count[digit] += 1
            
            # Calculate the contribution of this digit position to the total difference
            # For each digit, calculate how many pairs it forms with different digits
            for digit in range(10):
                count = digit_count[digit]
                # All pairs formed with this digit and other digits
                total_diff += count * (len(nums) - count)
        
        # Each pair is counted twice, so divide by 2
        return total_diff // 2