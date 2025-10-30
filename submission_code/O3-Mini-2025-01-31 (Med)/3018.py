class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        # Pointer for str2 we are trying to match
        j = 0
        n = len(str1)
        m = len(str2)
        
        # Iterate through str1 and try to match str2 as a subsequence.
        for i in range(n):
            # If we've matched all characters from str2, break early.
            if j == m:
                break
            
            # Check if str1[i] already equals the desired letter from str2.
            if str1[i] == str2[j]:
                j += 1
            # Otherwise, check if incrementing str1[i] cyclically gives the desired letter.
            elif ((ord(str1[i]) - ord('a') + 1) % 26) + ord('a') == ord(str2[j]):
                j += 1
        
        # We succeeded if j reached the end of str2.
        return j == m