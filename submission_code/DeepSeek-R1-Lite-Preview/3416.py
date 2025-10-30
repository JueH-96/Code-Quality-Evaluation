class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        # Determine the number of digits in the numbers
        D = len(str(nums[0]))
        N = len(nums)
        total_sum = 0
        # Precompute 10^p for faster digit extraction
        powers = [10 ** p for p in range(D)]
        for p in range(D):
            freq = [0] * 10
            # Extract the digit at position p for each number
            for num in nums:
                digit = (num // powers[p]) % 10
                freq[digit] += 1
            # Calculate the number of pairs with the same digit at position p
            same_pairs_p = sum(count * (count - 1) // 2 for count in freq)
            # Total number of pairs
            total_pairs = N * (N - 1) // 2
            # Number of pairs with different digits at position p
            diff_pairs_p = total_pairs - same_pairs_p
            # Add to the total sum
            total_sum += diff_pairs_p
        return total_sum