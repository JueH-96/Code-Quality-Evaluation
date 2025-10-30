class Solution:
    def minimizedStringLength(self, s: str) -> int:
        """
        Calculates the length of the minimized string by finding the number of
        unique characters.

        The operation described allows for the removal of duplicate characters.
        If a character 'c' appears more than once, we can always use one
        occurrence of 'c' as a pivot to remove another. This process can be
        repeated until only one instance of 'c' remains.

        Therefore, the length of the minimized string is equal to the number
        of unique characters in the original string.
        """
        
        # A set stores only unique elements. By converting the string to a set,
        # we get all the unique characters. The length of the set gives us the
        # count of unique characters, which is the minimized string length.
        return len(set(s))