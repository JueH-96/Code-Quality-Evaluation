class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        # Convert all nums to strings (all have the same number of digits)
        str_nums = list(map(str, nums))
        
        n = len(nums)            # Number of integers
        d = len(str_nums[0])     # Number of digits in each integer
        
        # Frequency array: freq[pos][digit] = how many numbers have 'digit' at position pos
        freq = [[0]*10 for _ in range(d)]
        
        # Collect frequencies
        for s in str_nums:
            for pos in range(d):
                digit = int(s[pos])
                freq[pos][digit] += 1
                
        # Calculate sum of digit differences
        total_sum = 0
        
        # Total number of pairs
        total_pairs = n * (n - 1) // 2
        
        for pos in range(d):
            # Count how many pairs have the same digit at this position
            same_pairs = 0
            for digit_count in freq[pos]:
                same_pairs += digit_count * (digit_count - 1) // 2
                
            # Pairs that differ at this position contribute 1 per pair
            differ_pairs = total_pairs - same_pairs
            total_sum += differ_pairs
            
        return total_sum