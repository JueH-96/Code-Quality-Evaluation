class Solution:
    def getEncryptedString(self, s: str, k: int) -> str:
        n = len(s)
        result = ""
        
        for i in range(n):
            # Calculate the index of the k^th character after the current character
            # in a cyclic manner
            index = (i + k) % n
            
            # Append the k^th character to the result
            result += s[index]
        
        return result