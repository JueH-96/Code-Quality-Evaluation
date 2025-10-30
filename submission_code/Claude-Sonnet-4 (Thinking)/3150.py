class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        beautiful_substrings = []
        
        # Find all beautiful substrings
        for i in range(len(s)):
            ones_count = 0
            for j in range(i, len(s)):
                if s[j] == '1':
                    ones_count += 1
                
                if ones_count == k:
                    beautiful_substrings.append(s[i:j+1])
                elif ones_count > k:
                    break
        
        if not beautiful_substrings:
            return ""
        
        # Find the shortest length
        min_length = min(len(sub) for sub in beautiful_substrings)
        
        # Return the lexicographically smallest among the shortest
        return min(sub for sub in beautiful_substrings if len(sub) == min_length)