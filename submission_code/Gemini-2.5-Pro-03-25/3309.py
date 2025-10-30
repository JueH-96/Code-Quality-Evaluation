from typing import List

class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        """
        Counts the number of pairs (i, j) such that i < j and words[i] is both a prefix and a suffix of words[j].

        Args:
            words: A list of strings.

        Returns:
            The count of such pairs.
        """

        n = len(words)
        count = 0

        # Define the helper function isPrefixAndSuffix
        def isPrefixAndSuffix(str1: str, str2: str) -> bool:
            """
            Checks if str1 is both a prefix and a suffix of str2.

            Args:
                str1: The potential prefix and suffix string.
                str2: The string to check against.

            Returns:
                True if str1 is both a prefix and suffix of str2, False otherwise.
            """
            # Optimization: if str1 is longer than str2, it cannot be a prefix or suffix
            if len(str1) > len(str2):
                return False
            
            # Check if str1 is a prefix of str2
            is_prefix = str2.startswith(str1)
            # Check if str1 is a suffix of str2
            is_suffix = str2.endswith(str1)
            
            # Return true only if both conditions are met
            return is_prefix and is_suffix

        # Iterate through all possible pairs (i, j) such that i < j
        for i in range(n):
            for j in range(i + 1, n):
                # Check if words[i] is a prefix and suffix of words[j]
                if isPrefixAndSuffix(words[i], words[j]):
                    count += 1
                    
        return count