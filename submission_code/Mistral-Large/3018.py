class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        # Iterate through each character in str2
        for char in str2:
            found = False
            # Check if the current character can be found in str1
            for i in range(len(str1)):
                if str1[i] == char or (str1[i] < char and str1[i] <= 'z' and char <= 'z'):
                    str1 = str1[i+1:]
                    found = True
                    break
            # If the character is not found, return False
            if not found:
                return False
        return True