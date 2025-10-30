class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        def is_subsequence(s1, s2):
            it = iter(s1)
            return all(char in it for char in s2)

        if is_subsequence(str1, str2):
            return True

        for i in range(len(str1)):
            new_char = chr((ord(str1[i]) - ord('a') + 1) % 26 + ord('a'))
            new_str1 = str1[:i] + new_char + str1[i+1:]
            if is_subsequence(new_str1, str2):
                return True

        return False