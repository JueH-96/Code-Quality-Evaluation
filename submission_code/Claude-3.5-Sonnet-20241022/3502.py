class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        result = 0
        
        # For each possible starting position
        for i in range(n):
            # Initialize character frequency counter
            char_count = {}
            
            # For each possible ending position starting from i
            for j in range(i, n):
                # Add current character to frequency counter
                char_count[s[j]] = char_count.get(s[j], 0) + 1
                
                # Check if any character appears at least k times
                for count in char_count.values():
                    if count >= k:
                        result += 1
                        break
        
        return result