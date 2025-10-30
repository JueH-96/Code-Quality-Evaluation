from typing import List

class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        """
        Calculates the sum of digit differences between all pairs of integers in nums.
        All integers in nums have the same number of digits.

        Args:
            nums: A list of positive integers with the same number of digits.

        Returns:
            The total sum of digit differences between all pairs.
        """
        n = len(nums)
        
        # Get the number of digits, d.
        # Since all numbers have the same number of digits, we can check the first one.
        # Convert the first number to a string and get its length.
        num_str = str(nums[0])
        d = len(num_str)
        
        total_diff = 0
        
        # Iterate through each digit position from right to left (units, tens, hundreds, ...)
        # Position k=0 is the units place (10^0).
        # Position k=1 is the tens place (10^1).
        # ...
        # Position k=d-1 is the leftmost place (10^(d-1)).
        # The loop runs d times, processing one digit position at a time.
        for k in range(d):
            # For the current digit position k, we need to count how many numbers
            # in the input list `nums` have each digit (0 through 9) at this specific position.
            # Initialize an array `counts` of size 10 to store these frequencies.
            # counts[digit] will store the number of times 'digit' appears at position k.
            counts = [0] * 10
            
            # Calculate the power of 10 corresponding to the current digit position k.
            # This is used to extract the digit at position k using integer division and modulo.
            # For k=0 (units), power_of_10 = 10^0 = 1.
            # For k=1 (tens), power_of_10 = 10^1 = 10.
            # For k=d-1, power_of_10 = 10^(d-1).
            power_of_10 = 10**k
            
            # Iterate through each number in the input list `nums`.
            for num in nums:
                # Extract the digit at position k (from the right, 0-indexed) for the current number `num`.
                # Integer division by 10^k shifts the number so the digit at position k becomes the units digit.
                # Taking modulo 10 extracts the units digit.
                digit = (num // power_of_10) % 10
                
                # Increment the count for the extracted digit at this position k.
                counts[digit] += 1
                
            # Now we have the counts of each digit (0-9) at position k across all `n` numbers.
            # counts[d] represents the number of integers in `nums` that have the digit 'd' at position k.
            # The sum of all counts must equal n: sum(counts) == n.
            
            # We want to find the number of pairs {i, j} with i < j such that the digits
            # at position k in nums[i] and nums[j] are different.
            # This is equivalent to Sum_{0 <= a < b <= 9} counts[a] * counts[b].
            # This sum can be calculated efficiently using the formula:
            # (n^2 - Sum_{d=0 to 9} counts[d]^2) // 2.
            # This formula comes from considering the total number of ordered pairs (i, j) with i != j (which is n*n - n = n*(n-1)).
            # The number of ordered pairs (i, j) with i != j where digits are the same is Sum_d counts[d] * (counts[d] - 1).
            # The number of ordered pairs (i, j) with i != j where digits are different is n*(n-1) - Sum_d counts[d]*(counts[d]-1) = n^2 - n - (Sum_d counts[d]^2 - Sum_d counts[d]).
            # Since Sum_d counts[d] = n, this becomes n^2 - n - Sum_d counts[d]^2 + n = n^2 - Sum_d counts[d]^2.
            # This counts ordered pairs (i, j) with i != j and different digits. Since diff(i, j) = diff(j, i),
            # the number of *unordered* pairs {i, j} with i != j and different digits is half of this: (n^2 - Sum_d counts[d]^2) // 2.
            
            # Calculate the sum of squares of the counts for each digit.
            sum_counts_sq = sum(count * count for count in counts)
            
            # Calculate the contribution to the total difference from the current position k.
            # This is the number of pairs with different digits at this position.
            # Using the derived formula. Integer division `// 2` is correct because the numerator is always even.
            position_diff = (n * n - sum_counts_sq) // 2
            
            # Add the difference contributed by this position to the overall total difference.
            total_diff += position_diff

        # After iterating through all digit positions, total_diff holds the sum of digit differences
        # for all pairs of integers in nums.
        return total_diff