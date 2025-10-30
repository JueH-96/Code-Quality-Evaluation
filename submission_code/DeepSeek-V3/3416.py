class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        from collections import defaultdict

        # Determine the number of digits in each number
        num_digits = len(str(nums[0]))
        n = len(nums)
        
        # Initialize a list of dictionaries to count the frequency of each digit at each position
        digit_counts = [defaultdict(int) for _ in range(num_digits)]
        
        # Populate the digit counts
        for num in nums:
            s = str(num)
            for i in range(num_digits):
                digit_counts[i][s[i]] += 1
        
        total = 0
        
        # Calculate the sum of digit differences for each position
        for i in range(num_digits):
            counts = digit_counts[i]
            # Total pairs is n * (n-1) / 2
            total_pairs = n * (n - 1) // 2
            # Sum of counts of each digit squared
            sum_squares = sum(v * (v - 1) // 2 for v in counts.values())
            # The number of pairs with the same digit is sum_squares
            # So the number of pairs with different digits is total_pairs - sum_squares
            total += (total_pairs - sum_squares)
        
        return total