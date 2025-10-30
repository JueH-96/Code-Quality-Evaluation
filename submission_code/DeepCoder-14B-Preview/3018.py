class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        i = j = 0
        while i < len(str1) and j < len(str2):
            s2_char = str2[j]
            # Calculate the two possible target characters in str1
            target1 = s2_char
            target2 = chr(((ord(s2_char) - ord('a')) - 1) % 26 + ord('a'))
            if str1[i] == target1 or str1[i] == target2:
                j += 1
            i += 1
        return j == len(str2)