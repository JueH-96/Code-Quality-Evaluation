class Solution:
    def shortestMatchingSubstring(self, s: str, p: str) -> int:
        # Find the indices of the two '*' characters in the pattern
        star1, star2 = -1, -1
        for i, char in enumerate(p):
            if char == '*':
                if star1 == -1:
                    star1 = i
                else:
                    star2 = i
        
        # Initialize the minimum length to a large value
        min_len = float('inf')
        
        # Iterate through the string s
        for i in range(len(s)):
            for j in range(i, len(s)):
                # Check if the substring s[i:j+1] matches the pattern p
                if self.matches(s[i:j+1], p, star1, star2):
                    min_len = min(min_len, j - i + 1)
        
        # If no matching substring was found, return -1
        return min_len if min_len != float('inf') else -1
    
    def matches(self, substring: str, pattern: str, star1: int, star2: int) -> bool:
        # Check if the substring matches the pattern
        i, j = 0, 0
        while i < len(substring) and j < len(pattern):
            if pattern[j] == '*':
                if j == star1:
                    # Match any sequence of zero or more characters
                    return True
                else:
                    # Match any character
                    j += 1
            elif substring[i] == pattern[j]:
                i += 1
                j += 1
            else:
                return False
        
        # Check if the entire pattern has been matched
        return j == len(pattern)