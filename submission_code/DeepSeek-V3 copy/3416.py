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
        # Calculate the total digit differences
        for i in range(num_digits):
            counts = digit_counts[i]
            # Calculate the total pairs
            total_pairs = n * (n - 1) // 2
            # Calculate the sum of pairs where the digits are the same
            same_pairs = 0
            for digit in counts:
                cnt = counts[digit]
                same_pairs += cnt * (cnt - 1) // 2
            # The digit difference is total_pairs - same_pairs
            total += (total_pairs - same_pairs)
        
        return total