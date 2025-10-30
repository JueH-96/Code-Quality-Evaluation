class Solution:
    def calculateScore(self, s: str) -> int:
        # Set to keep track of marked indices
        marked = set()
        total_score = 0
        
        for i in range(len(s)):
            # Skip if this index is already marked
            if i in marked:
                continue
            
            # Calculate the mirror character
            # 'a' (ASCII 97) maps to 'z' (ASCII 122)
            # 'b' (ASCII 98) maps to 'y' (ASCII 121), etc.
            # We can use the formula: mirror = chr(219 - ord(char))
            mirror_char = chr(219 - ord(s[i]))
            
            # Find the closest unmarked index j < i such that s[j] is the mirror of s[i]
            for j in range(i-1, -1, -1):
                if j not in marked and s[j] == mirror_char:
                    # Mark both indices
                    marked.add(i)
                    marked.add(j)
                    # Add distance to score
                    total_score += i - j
                    break
        
        return total_score