class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        # Initialize pointers for str1 and str2
        i = 0  # pointer for str1
        j = 0  # pointer for str2

        # Iterate through str1 to find characters matching str2 in order
        while i < len(str1) and j < len(str2):
            char1 = str1[i]
            char2 = str2[j]

            # Calculate the cyclically incremented character of char1
            # 'a' -> 'b', ..., 'y' -> 'z', 'z' -> 'a'
            next_char1 = chr(ord(char1) + 1) if char1 != 'z' else 'a'

            # Check if str1[i] (original or incremented) can match str2[j]
            # If char1 == char2, we can potentially use str1[i] as is.
            # If next_char1 == char2, we can potentially use str1[i] after incrementing.
            if char1 == char2 or next_char1 == char2:
                # If a match is found using either the original or incremented character,
                # it means str1[i] can serve as str2[j] in the subsequence.
                # Move to the next character in str2 that we need to find.
                j += 1

            # Always move to the next character in str1.
            # We consider str1[i] at most once for matching str2[j].
            i += 1

        # If the pointer j has reached the length of str2, it means we have found
        # all characters of str2 as a subsequence within str1, by potentially
        # incrementing characters in str1 where needed.
        return j == len(str2)