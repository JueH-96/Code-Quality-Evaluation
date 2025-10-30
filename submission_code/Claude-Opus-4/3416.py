class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Convert all numbers to strings
        str_nums = [str(num) for num in nums]
        num_digits = len(str_nums[0])
        
        total_diff = 0
        
        # For each digit position
        for pos in range(num_digits):
            # Count frequency of each digit at this position
            digit_count = {}
            for num_str in str_nums:
                digit = num_str[pos]
                digit_count[digit] = digit_count.get(digit, 0) + 1
            
            # Calculate contribution of this position
            # For each pair of numbers, they contribute 1 if digits differ
            # Total pairs = n * (n-1) / 2
            # Pairs with same digit = sum of (count * (count-1) / 2) for each digit
            same_pairs = 0
            for count in digit_count.values():
                same_pairs += count * (count - 1) // 2
            
            different_pairs = n * (n - 1) // 2 - same_pairs
            total_diff += different_pairs
        
        return total_diff