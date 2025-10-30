class Solution:
    def isBalanced(self, num: str) -> bool:
        """
        Checks if a numeric string is 'balanced' – the sum of the digits at even
        indices equals the sum at odd indices (0-based indexing).

        Parameters
        ----------
        num : str
            The string consisting solely of digit characters.

        Returns
        -------
        bool
            True if the string is balanced, otherwise False.
        """
        even_sum = 0
        odd_sum = 0
        
        for idx, ch in enumerate(num):
            digit = ord(ch) - 48  # Faster than int(ch) for single characters
            if idx & 1:           # odd index (bitwise check)
                odd_sum += digit
            else:                 # even index
                even_sum += digit
        
        return even_sum == odd_sum