class Solution:
    def isFascinating(self, n: int) -> bool:
        """
        Checks if a 3-digit number n is fascinating.
        A number n is fascinating if concatenating n, 2*n, and 3*n results in a number
        that contains all digits from 1 to 9 exactly once and does not contain any 0s.

        Args:
            n: An integer between 100 and 999, inclusive, guaranteed to have exactly 3 digits.

        Returns:
            True if n is fascinating, False otherwise.
        """
        
        # Calculate 2*n and 3*n
        n2 = 2 * n
        n3 = 3 * n

        # Concatenate n, 2*n, and 3*n into a single string
        # The input n is guaranteed to be 3 digits (100 <= n <= 999).
        # 2*n will be between 200 and 1998 (3 or 4 digits).
        # 3*n will be between 300 and 2997 (3 or 4 digits).
        s = str(n) + str(n2) + str(n3)
        
        # The condition for being fascinating requires the concatenated number to contain
        # all digits from 1 to 9 exactly once and no 0s.
        # This implies the concatenated string must be a permutation of "123456789".
        
        # We can check this efficiently by sorting the characters of the concatenated string
        # and comparing the result to the string "123456789".
        
        # This single comparison implicitly checks all necessary conditions:
        # 1. Length Check: If the length of s is not 9, the sorted string cannot equal "123456789".
        # 2. No Zeros Check: If s contains '0', the sorted string will start with '0' (or contain '0') and won't equal "123456789".
        # 3. All Digits 1-9 Check: If s has the correct length (9) and no zeros, sorting it will produce "123456789" if and only if it contains each digit from 1 to 9 exactly once. 
        #    If any digit is missing or repeated, the sorted string will differ.
        
        return "".join(sorted(s)) == "123456789"