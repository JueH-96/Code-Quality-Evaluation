class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        n = len(nums)
        # Per constraints, n >= 2, so no need for an n < 2 check.

        # All numbers in nums are positive and have the same number of digits.
        # We can get the number of digits from the first number.
        num_digits = len(str(nums[0]))
        
        total_diff = 0
        
        # To avoid modifying the original list, we work on a copy.
        # This allows us to process digits from right to left by repeatedly
        # taking the number modulo 10 and then integer dividing by 10.
        temp_nums = list(nums)
        
        for _ in range(num_digits):
            # For the current digit position, count the frequency of each digit (0-9).
            counts = [0] * 10
            
            for i in range(n):
                # Get the last digit (units digit of the current number).
                digit = temp_nums[i] % 10
                counts[digit] += 1
                # Remove the last digit to process the next digit position in the subsequent iteration.
                temp_nums[i] //= 10

            # For this position, calculate the number of pairs with different digits.
            # If a digit appears `c` times, it forms `c * (n - c)` pairs with numbers
            # that have a different digit at this position.
            # Summing this over all digits double-counts each differing pair.
            diff_pairs_at_pos = 0
            for c in counts:
                diff_pairs_at_pos += c * (n - c)
            
            # Divide by 2 to correct for the double-counting.
            total_diff += diff_pairs_at_pos // 2
            
        return total_diff