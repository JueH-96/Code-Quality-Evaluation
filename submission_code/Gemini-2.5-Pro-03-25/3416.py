import math # This import is not strictly needed if using len(str()) but kept for completeness if alternative digit count methods were considered.
from typing import List

class Solution:
  """
  Solves the problem of finding the sum of digit differences between all pairs of integers in a list,
  where all integers have the same number of digits.

  The core idea is to iterate through each digit position (e.g., ones, tens, hundreds)
  and calculate the contribution of that position to the total sum of differences.
  For a fixed position, we count how many pairs of numbers have different digits at that position.
  Summing these counts over all positions gives the final answer.
  """
  def sumDigitDifferences(self, nums: List[int]) -> int:
    """
    Calculates the sum of digit differences between all pairs of integers in nums.

    Args:
      nums: A list of positive integers, all guaranteed to have the same number of digits.
            Constraints: 2 <= len(nums) <= 10^5, 1 <= nums[i] < 10^9.

    Returns:
      The total sum of digit differences between all pairs in nums.
    """
    n = len(nums)
    # Constraints state 2 <= n, so we don't need to handle n < 2 explicitly.

    # Constraints state all integers in nums have the same number of digits.
    # We determine the number of digits (d) using the first number.
    # Constraints state 1 <= nums[i] < 10^9, so numbers have between 1 and 9 digits.
    if nums[0] == 0:
        # This case should not occur based on the constraint nums[i] >= 1.
        # If 0 were possible and had the same number of digits (e.g., [00, 01]),
        # we'd need clarification. Assuming positive integers as stated.
        # Defaulting to 1 digit for 0 if it were allowed.
        d = 1 
    else:
        # Using string conversion is a reliable way to get the number of digits.
        num_str = str(nums[0])
        d = len(num_str)
        # Alternative (requires math import): d = math.floor(math.log10(nums[0])) + 1

    total_difference = 0
    # This variable represents the positional value (1, 10, 100, ...)
    # corresponding to the current digit being processed (from right to left).
    power_of_10 = 1 

    # Iterate through each digit position k, from 0 (ones place) up to d-1 (most significant digit).
    for k in range(d):
        # Initialize frequency counts for digits 0 through 9 at the current position k.
        # counts[dg] will store how many numbers in nums have the digit 'dg' at position k.
        counts = [0] * 10 

        # Count the occurrences of each digit (0-9) at the current position k across all numbers.
        for num in nums:
            # Extract the digit at the k-th position (from the right, 0-indexed)
            # using integer division and the modulo operator.
            # Example: for num=123, k=0 (power=1), digit = (123 // 1) % 10 = 3
            #          for num=123, k=1 (power=10), digit = (123 // 10) % 10 = 2
            #          for num=123, k=2 (power=100), digit = (123 // 100) % 10 = 1
            digit = (num // power_of_10) % 10
            counts[digit] += 1

        # Calculate the contribution of this specific digit position (k) to the total sum of differences.
        # Logic:
        # Consider a fixed digit position k. Let's count the pairs (i, j) with i < j such that
        # the digit of nums[i] at position k is different from the digit of nums[j] at position k.
        #
        # An efficient way to count this:
        # For a specific digit value `dg` (0-9), suppose it appears `count = counts[dg]` times at position k.
        # These `count` numbers all have the same digit `dg` at this position.
        # The remaining `n - count` numbers have a different digit at this position.
        # Each of the `count` numbers (with digit `dg`) forms a differing pair with each of the `n - count` numbers.
        # So, the number of differing pairs involving numbers with digit `dg` is `count * (n - count)`.
        #
        # Summing `count * (n - count)` over all possible digit values (dg = 0 to 9) counts
        # every differing pair exactly twice (once from the perspective of the first number in the pair,
        # and once from the perspective of the second number).
        # Example: Pair (A, B) where digit_k(A) != digit_k(B).
        #          This pair is counted when considering digit_k(A) and when considering digit_k(B).
        
        position_contribution_doubled = 0
        # We iterate through the counts array (indices 0 to 9).
        for count in counts:
             # Add the contribution for this digit value.
             # count * (n - count) represents the number of pairs formed by numbers with this digit
             # and numbers with a different digit at this position.
             position_contribution_doubled += count * (n - count)

        # The sum `position_contribution_doubled` counts each differing pair twice.
        # Divide by 2 to get the actual number of differing pairs for this position k.
        # This division results in an integer because `position_contribution_doubled` is always even.
        # (Proof: Sum[c*(n-c)] = n*Sum[c] - Sum[c*c] = n*n - Sum[c*c]. n*n and Sum[c*c] have the same parity.)
        total_difference += position_contribution_doubled // 2

        # Prepare for the next digit position by multiplying power_of_10 by 10.
        # This check avoids an unnecessary multiplication after processing the last digit (k=d-1).
        if k < d - 1:
            power_of_10 *= 10
            # Note: Python's arbitrary precision integers handle large power_of_10 values.
            # In languages with fixed-size integers, overflow checks might be needed if d were very large.
            # Given constraints (nums[i] < 10^9, d <= 9), power_of_10 goes up to 10^8, well within standard 64-bit integer limits.

    # After iterating through all digit positions, total_difference holds the sum over all positions.
    return total_difference