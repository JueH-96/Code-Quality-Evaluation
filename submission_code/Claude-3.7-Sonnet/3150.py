class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        min_len = float('inf')
        beautiful_substrings = []
        
        for i in range(n):
            count_ones = 0
            for j in range(i, n):
                if s[j] == '1':
                    count_ones += 1
                
                if count_ones == k:
                    length = j - i + 1
                    if length < min_len:
                        min_len = length
                        beautiful_substrings = [s[i:j+1]]
                    elif length == min_len:
                        beautiful_substrings.append(s[i:j+1])
                    break
        
        if not beautiful_substrings:
            return ""
        
        # Return the lexicographically smallest substring
        return min(beautiful_substrings)