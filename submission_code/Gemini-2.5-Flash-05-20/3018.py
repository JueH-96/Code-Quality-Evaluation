class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        p1 = 0  # Pointer for str1
        p2 = 0  # Pointer for str2

        len1 = len(str1)
        len2 = len(str2)

        # If str2 is empty, it's always a subsequence of any string.
        if len2 == 0:
            return True
        
        # If str1 is shorter than str2, str2 cannot be a subsequence of str1.
        if len1 < len2:
            return False

        # Iterate through str1 using p1 and try to match characters of str2 using p2.
        while p1 < len1 and p2 < len2:
            char1_val = ord(str1[p1]) - ord('a')
            char2_val = ord(str2[p2]) - ord('a')

            # Check if str1[p1] can match str2[p2] directly OR after one cyclic increment.
            # 1. Direct match: str1[p1] is the same as str2[p2].
            # 2. Cyclic increment match: (str1[p1] + 1) cyclically results in str2[p2].
            if char1_val == char2_val or ((char1_val + 1) % 26) == char2_val:
                p2 += 1  # If a match is found, move to the next character in str2.
            
            p1 += 1  # Always move to the next character in str1 to find the next potential match.

        # If p2 has reached the end of str2, it means we successfully found all characters
        # of str2 as a subsequence within str1, respecting the transformation rules.
        return p2 == len2