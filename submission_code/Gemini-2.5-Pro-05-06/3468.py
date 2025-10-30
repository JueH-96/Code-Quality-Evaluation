class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        
        # The problem asks to replace each character s[i] with the k-th character
        # after s[i] in a cyclic manner.
        # The character s[i] is at index i.
        # The k-th character after s[i] is at index (i + k) % n.
        # So, the i-th character of the encrypted string, let's call it e[i],
        # will be s[(i + k) % n].
        #
        # This transformation is equivalent to a left cyclic shift of the string s
        # by k positions.
        
        # First, calculate the effective shift amount.
        # Since the string length is n, shifting by k is the same as shifting
        # by k % n. This ensures the shift amount is in the range [0, n-1].
        effective_k = k % n
        
        # A left cyclic shift by 'effective_k' positions can be performed by
        # taking the part of the string from 'effective_k' to the end,
        # and concatenating it with the part of the string from the beginning
        # up to 'effective_k'.
        # s_part_2 = s[effective_k:]  (characters from index effective_k to n-1)
        # s_part_1 = s[:effective_k] (characters from index 0 to effective_k-1)
        # The shifted string is s_part_2 + s_part_1.
        
        # Example: s = "dart", k = 3
        # n = 4
        # effective_k = 3 % 4 = 3
        # s[3:] = "t"
        # s[:3] = "dar"
        # Result = "t" + "dar" = "tdar" (Matches Example 1)

        # Example: s = "aaa", k = 1
        # n = 3
        # effective_k = 1 % 3 = 1
        # s[1:] = "aa"
        # s[:1] = "a"
        # Result = "aa" + "a" = "aaa" (Matches Example 2)
        
        # If effective_k is 0 (e.g., k is a multiple of n), then:
        # s[0:] is the original string s.
        # s[:0] is an empty string.
        # The result is s + "" = s, meaning the string is unchanged, which is correct.
        
        return s[effective_k:] + s[:effective_k]