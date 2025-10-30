class Solution:
    def generateString(self, str1: str, str2: str) -> str:
        n = len(str1)
        m = len(str2)
        
        # Initialize the result array with '?' to indicate unknown characters
        result = ['?'] * (n + m - 1)
        
        for i in range(n):
            if str1[i] == 'T':
                # If str1[i] is 'T', we must set the substring to str2
                for j in range(m):
                    if result[i + j] == '?':
                        result[i + j] = str2[j]
                    elif result[i + j] != str2[j]:
                        return ""  # Conflict detected
            else:  # str1[i] == 'F'
                # If str1[i] is 'F', we must ensure the substring is not str2
                for j in range(m):
                    if result[i + j] == '?':
                        continue  # No conflict yet
                    elif result[i + j] == str2[j]:
                        return ""  # Conflict detected
        
        # Fill in the remaining '?' with the smallest lexicographical character
        for i in range(len(result)):
            if result[i] == '?':
                result[i] = 'a'  # Fill with 'a' as the smallest character
        
        return ''.join(result)