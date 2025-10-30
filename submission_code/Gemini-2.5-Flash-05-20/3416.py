from typing import List

class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Convert all numbers to strings once for efficient digit access.
        # This allows us to access digits by index (position).
        # Example: [13, 23, 12] becomes ["13", "23", "12"]
        # This step has a time complexity of O(N * D), where D is the number of digits.
        s_nums = [str(num) for num in nums]
        
        # All integers in nums are guaranteed to have the same number of digits.
        # We can determine the number of digits from the first number's string representation.
        num_digits = len(s_nums[0])
        
        total_digit_differences = 0
        
        # Iterate through each digit position, from the most significant to the least significant.
        # For example, if D=2, k will be 0 (tens place) and 1 (units place).
        # This outer loop runs D times.
        for k in range(num_digits):
            # `count_for_position` will store the frequency of each digit (0-9)
            # at the current digit position `k` across all numbers in `nums`.
            # Example: For k=0 in ["13", "23", "12"], count_for_position[1]=2, count_for_position[2]=1.
            count_for_position = [0] * 10 
            
            # Populate the counts for the current digit position `k`.
            # This inner loop runs N times for each digit position.
            for s_num in s_nums:
                digit = int(s_num[k])  # Extract the digit at position k
                count_for_position[digit] += 1 # Increment its frequency
            
            # Calculate the number of pairs that have *different* digits at the current position `k`.
            # For a digit `d` appearing `count` times at position `k`:
            # Any of these `count` numbers will have a different digit at position `k`
            # when paired with any of the `(n - count)` numbers that *do not* have digit `d` at position `k`.
            # So, `count * (n - count)` represents the number of ordered pairs where one number has digit `d`
            # and the other has a different digit at position `k`.
            
            diff_at_current_position = 0
            for digit_count in count_for_position:
                # We only need to process digits that actually appeared at this position (digit_count > 0).
                # Although `0 * (n - 0)` is 0, so the condition isn't strictly necessary for correctness,
                # it can slightly optimize by skipping zero multiplications.
                if digit_count > 0:
                    diff_at_current_position += digit_count * (n - digit_count)
            
            # `diff_at_current_position` currently sums up `count * (n - count)` for all digits.
            # This sum counts each unique pair (num_i, num_j) where digits differ at position `k` twice:
            # - Once when `num_i`'s digit is considered (and it's paired with `num_j`).
            # - Once when `num_j`'s digit is considered (and it's paired with `num_i`).
            # Therefore, to get the count of unique pairs, we divide by 2.
            # Integer division `//` is used as the sum will always be an even number.
            total_digit_differences += diff_at_current_position // 2
            
        return total_digit_differences