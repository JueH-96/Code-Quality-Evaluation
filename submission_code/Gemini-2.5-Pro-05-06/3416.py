from typing import List

class Solution:
  def sumDigitDifferences(self, nums: List[int]) -> int:
    n = len(nums)
    
    # Constraints: 2 <= nums.length <= 10^5.
    # Constraints: 1 <= nums[i] < 10^9. All positive.
    # Constraints: All integers in nums have the same number of digits.
    
    # Determine the number of digits, d.
    # Since nums[0] >= 1, str(nums[0]) is safe and gives correct length.
    s_num0 = str(nums[0])
    num_digits = len(s_num0)

    # This case should not be reached given constraints (num_digits >= 1 because nums[i] >= 1)
    if num_digits == 0:
        return 0

    total_digit_difference_sum = 0
    
    # Create a mutable copy of nums to extract digits by repeated %10 and //=10 operations.
    # This is generally efficient for extracting digits from right to left.
    current_nums_for_processing = list(nums) 
    
    # Iterate for each digit position, from rightmost (units place) to leftmost.
    # The loop runs 'num_digits' times.
    for _ in range(num_digits): # _ indicates loop variable is not used directly for indexing
        # For the current digit position, count frequencies of each digit (0-9)
        # across all n numbers.
        counts_at_this_pos = [0] * 10 # counts_at_this_pos[d_val] = frequency of digit d_val
        
        for i in range(n):
            # Extract the last (rightmost) digit of current_nums_for_processing[i]
            digit = current_nums_for_processing[i] % 10
            counts_at_this_pos[digit] += 1
            # Remove the last digit, effectively shifting the number one place to the right
            # This prepares it for the next iteration of the outer loop (next digit position).
            current_nums_for_processing[i] //= 10
        
        # Calculate contribution to the total sum from this specific digit position.
        # The formula sum(c_D * (n - c_D)) / 2 is used here.
        # c_D is the frequency of a digit D (k_digit_frequency in code).
        # n - c_D is the count of numbers that have a different digit at this position.
        # The product c_D * (n - c_D) sums these possibilities for all numbers with digit D.
        # Summing this over all D (0-9) counts each differing pair twice.
        
        position_contribution_numerator = 0
        for k_digit_frequency in counts_at_this_pos: # k_digit_frequency is c_D
            position_contribution_numerator += k_digit_frequency * (n - k_digit_frequency)
        
        # Each pair (num_a, num_b) with different digits at this position is counted twice
        # in position_contribution_numerator. So, divide by 2.
        total_digit_difference_sum += position_contribution_numerator // 2
            
    return total_digit_difference_sum