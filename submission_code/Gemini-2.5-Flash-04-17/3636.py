class Solution:
    def isBalanced(self, num: str) -> bool:
        """
        Checks if a string of digits is balanced.
        A string is balanced if the sum of digits at even indices equals the sum of digits at odd indices.

        Args:
            num: A string consisting of only digits.

        Returns:
            True if num is balanced, False otherwise.
        """
        even_sum = 0
        odd_sum = 0

        for i in range(len(num)):
            digit = int(num[i]) # Convert character digit to integer

            if i % 2 == 0: # Even index
                even_sum += digit
            else: # Odd index
                odd_sum += digit

        return even_sum == odd_sum