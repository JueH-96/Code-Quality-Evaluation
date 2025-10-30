class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        """
        Determines if str2 can be made a subsequence of str1 by incrementing
        characters of str1 at most once.
        """
        
        # i is a pointer for str1, and j is a pointer for str2.
        i, j = 0, 0
        n1, n2 = len(str1), len(str2)

        # We iterate through str1 with pointer i, trying to find characters
        # that can form the subsequence str2, which is tracked by pointer j.
        while i < n1 and j < n2:
            # For each character str1[i], we check if it can match str2[j].
            # A match is possible in two ways:
            # 1. str1[i] is already the character we need (str2[j]).
            # 2. str1[i] can be incremented to become str2[j].

            # The next character in the cycle ('z' becomes 'a').
            next_char = 'a' if str1[i] == 'z' else chr(ord(str1[i]) + 1)

            # If a match is found (either with the original char or the incremented one)
            if str1[i] == str2[j] or next_char == str2[j]:
                # We have successfully matched the j-th character of str2.
                # Now, we need to look for the (j+1)-th character.
                j += 1
            
            # We always advance the pointer in str1. This is because we are looking
            # for a subsequence, so characters must be picked in order
            # from str1. Once a character at index i is considered (used or not),
            # we must move on to characters at indices > i.
            i += 1
        
        # After iterating through str1, if j has reached the length of str2,
        # it means we have found a matching character in str1 for every character
        # in str2, in the correct order. Thus, str2 can be formed as a subsequence.
        return j == n2