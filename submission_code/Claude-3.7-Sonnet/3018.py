class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        i, j = 0, 0  # Pointers for str1 and str2
        
        while i < len(str1) and j < len(str2):
            # Check if current characters match directly or after one increment
            if str1[i] == str2[j] or self.canTransform(str1[i], str2[j]):
                j += 1  # Match found, move to next character in str2
            i += 1  # Always move to next character in str1
        
        return j == len(str2)  # If all characters in str2 were matched
    
    def canTransform(self, char1: str, char2: str) -> bool:
        # Check if char1 can be transformed to char2 with one increment
        # a->b, b->c, ..., z->a
        return (ord(char2) - ord(char1)) % 26 == 1