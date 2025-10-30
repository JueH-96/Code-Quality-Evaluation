class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        # Initialize two pointers for str1 and str2
        i, j = 0, 0
        
        # Initialize a set to store the indices of str1 that have been incremented
        incremented = set()
        
        # Iterate through str2
        while j < len(str2):
            # If the current character in str2 is not found in the remaining characters of str1
            if i == len(str1):
                return False
            
            # If the current character in str1 is equal to the current character in str2
            if str1[i] == str2[j]:
                # Move to the next character in both str1 and str2
                i += 1
                j += 1
            # If the current character in str1 is one character before the current character in str2
            elif ord(str1[i]) + 1 == ord(str2[j]):
                # If the current index in str1 has not been incremented before
                if i not in incremented:
                    # Add the current index to the set of incremented indices
                    incremented.add(i)
                    # Move to the next character in str2
                    j += 1
                # Move to the next character in str1
                i += 1
            # If the current character in str1 is not equal to the current character in str2 and is not one character before it
            else:
                # Move to the next character in str1
                i += 1
        
        # If all characters in str2 have been found in str1, return True
        return True