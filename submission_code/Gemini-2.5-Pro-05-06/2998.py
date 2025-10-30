class Solution:
    def _is_symmetric(self, num: int) -> bool:
        """
        Helper method to check if a number is symmetric.
        """
        s_num = str(num)
        n_digits = len(s_num)

        # Condition 1: Must have an even number of digits.
        # Numbers with an odd number of digits are never symmetric.
        if n_digits % 2 != 0:
            return False

        # At this point, n_digits is even.
        # 'n' in the problem description "2 * n digits" is half_len here.
        half_len = n_digits // 2
        
        first_half_str = s_num[:half_len]
        second_half_str = s_num[half_len:]

        # Condition 2: Sum of first n digits equals sum of last n digits.
        # Calculate sum of digits for the first half.
        sum_first_half = 0
        for digit_char in first_half_str:
            sum_first_half += int(digit_char)
        
        # Calculate sum of digits for the second half.
        sum_second_half = 0
        for digit_char in second_half_str:
            sum_second_half += int(digit_char)
            
        return sum_first_half == sum_second_half

    def countSymmetricIntegers(self, low: int, high: int) -> int:
        """
        Counts the number of symmetric integers in the range [low, high].
        """
        count = 0
        for current_num in range(low, high + 1):
            if self._is_symmetric(current_num):
                count += 1
        return count