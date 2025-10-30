class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        count = 0
        
        # Iterate over all possible starting points of substrings
        for start in range(n):
            char_count = {}
            # Iterate over all possible ending points of substrings
            for end in range(start, n):
                char_count[s[end]] = char_count.get(s[end], 0) + 1
                
                # Check if any character appears at least k times
                if any(count >= k for count in char_count.values()):
                    count += 1
        
        return count