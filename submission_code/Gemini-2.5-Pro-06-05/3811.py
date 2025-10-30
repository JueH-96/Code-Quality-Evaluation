class Solution:
    def reverseDegree(self, s: str) -> int:
        """
        Calculates the reverse degree of a string.

        The reverse degree is the sum of products for each character.
        Each product is the character's value in the reversed alphabet
        ('a'=26, ..., 'z'=1) multiplied by its 1-indexed position in the string.

        Args:
            s: The input string, containing only lowercase English letters.

        Returns:
            The reverse degree of the string.
        """

        # We can solve this concisely using a generator expression with sum().
        # For each character and its 0-indexed position `i` from enumerate(s):
        # 1. Calculate the reversed alphabet value: 26 - (ord(char) - ord('a'))
        # 2. Get the 1-indexed string position: i + 1
        # 3. Multiply these two values.
        # The sum() function then totals these products.
        return sum((26 - (ord(char) - ord('a'))) * (i + 1) for i, char in enumerate(s))