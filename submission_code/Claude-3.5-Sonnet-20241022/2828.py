class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        result = list(s)
        original = s
        
        # Helper function to get previous character
        def getPrev(c):
            return chr(ord('z') if c == 'a' else ord(c) - 1)
            
        # Try all possible substrings
        for i in range(n):
            for j in range(i, n):
                # Create a copy of original string
                curr = list(s)
                
                # Replace substring with previous characters
                for k in range(i, j + 1):
                    curr[k] = getPrev(curr[k])
                
                # Convert list to string
                curr_str = ''.join(curr)
                
                # Update result if current string is lexicographically smaller
                if curr_str < ''.join(result):
                    result = list(curr_str)
        
        return ''.join(result)