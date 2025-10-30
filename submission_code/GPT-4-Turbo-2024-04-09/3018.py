class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        # Pointer for str2
        j = 0
        # Length of str2
        n = len(str2)
        
        # Iterate over each character in str1
        for i in range(len(str1)):
            # Check if current pointer j of str2 is less than its length and
            # the current character of str1 can be transformed to match str2[j]
            if j < n and (str1[i] == str2[j] or (ord(str2[j]) - ord(str1[i]) + 26) % 26 == 1):
                # Move to the next character in str2
                j += 1
            # If all characters of str2 are matched
            if j == n:
                return True
        
        # If not all characters of str2 are matched
        return False