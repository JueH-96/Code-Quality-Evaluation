class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Convert all numbers to strings and find the number of digits
        str_nums = [str(num) for num in nums]
        digit_len = len(str_nums[0])
        
        total_diff = 0
        
        # For each position
        for pos in range(digit_len):
            digit_counts = [0] * 10  # Counts for digits 0-9 at current position
            
            # Count digits at current position
            for s in str_nums:
                digit = int(s[pos])
                digit_counts[digit] += 1
            
            # Compute the number of pairs with different digits at this position
            same_digit_pairs = sum(count * (count - 1) // 2 for count in digit_counts)
            total_pairs = n * (n - 1) // 2
            pos_diff = total_pairs - same_digit_pairs
            
            total_diff += pos_diff
        
        return total_diff