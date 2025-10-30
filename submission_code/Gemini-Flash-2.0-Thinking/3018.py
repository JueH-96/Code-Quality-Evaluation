class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        n1 = len(str1)
        n2 = len(str2)

        def is_subsequence(s1, s2):
            i = 0
            j = 0
            while i < len(s1) and j < len(s2):
                if s1[i] == s2[j]:
                    j += 1
                i += 1
            return j == len(s2)

        if is_subsequence(str1, str2):
            return True

        # Try performing the operation once
        for i in range(n1):
            original_char = str1[i]
            modified_char_ord = (ord(original_char) - ord('a') + 1) % 26 + ord('a')
            modified_char = chr(modified_char_ord)
            modified_str1_single = list(str1)
            modified_str1_single[i] = modified_char
            if is_subsequence("".join(modified_str1_single), str2):
                return True

        # Try performing the operation on two indices
        for i in range(n1):
            for j in range(i, n1):
                modified_str1_multi = list(str1)

                original_char_i = str1[i]
                modified_char_ord_i = (ord(original_char_i) - ord('a') + 1) % 26 + ord('a')
                modified_char_i = chr(modified_char_ord_i)
                modified_str1_multi[i] = modified_char_i

                if i != j:
                    original_char_j = str1[j]
                    modified_char_ord_j = (ord(original_char_j) - ord('a') + 1) % 26 + ord('a')
                    modified_char_j = chr(modified_char_ord_j)
                    modified_str1_multi[j] = modified_char_j
                elif i == j:
                     pass # Only increment once

                if is_subsequence("".join(modified_str1_multi), str2):
                    return True

        return False