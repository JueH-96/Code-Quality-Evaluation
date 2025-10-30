class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        min_len = float('inf')
        result = ""
        
        # Try each starting position
        for i in range(n):
            count = 0
            # Expand from this starting position
            for j in range(i, n):
                if s[j] == '1':
                    count += 1
                
                if count == k:
                    # Found a beautiful substring
                    length = j - i + 1
                    substring = s[i:j+1]
                    
                    if length < min_len:
                        # Found a shorter beautiful substring
                        min_len = length
                        result = substring
                    elif length == min_len and substring < result:
                        # Found an equally short but lexicographically smaller substring
                        result = substring
                    break  # No need to extend further from this starting position
                elif count > k:
                    # Too many ones, no point extending further
                    break
        
        return result