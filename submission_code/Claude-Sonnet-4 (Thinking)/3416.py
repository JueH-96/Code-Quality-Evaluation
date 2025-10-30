class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return 0
        
        # All numbers have the same number of digits
        num_digits = len(str(nums[0]))
        
        total = 0
        
        # For each digit position
        for pos in range(num_digits):
            # Count frequency of each digit (0-9) at this position
            freq = [0] * 10
            
            for num in nums:
                digit = (num // (10 ** pos)) % 10
                freq[digit] += 1
            
            # Calculate how many pairs have different digits at this position
            pairs_with_same_digit = sum(f * (f - 1) // 2 for f in freq)
            total_pairs = n * (n - 1) // 2
            pairs_with_different_digits = total_pairs - pairs_with_same_digit
            
            total += pairs_with_different_digits
        
        return total