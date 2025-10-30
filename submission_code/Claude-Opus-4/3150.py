class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        min_len = float('inf')
        result = ""
        
        # Try all possible starting positions
        for i in range(n):
            count = 0
            # Expand window from position i
            for j in range(i, n):
                if s[j] == '1':
                    count += 1
                
                # When we have exactly k ones
                if count == k:
                    current_len = j - i + 1
                    current_substring = s[i:j+1]
                    
                    # Update if we found a shorter substring
                    if current_len < min_len:
                        min_len = current_len
                        result = current_substring
                    # If same length, choose lexicographically smaller
                    elif current_len == min_len:
                        if current_substring < result:
                            result = current_substring
                    break
                
                # If we exceed k ones, no point continuing from this start
                elif count > k:
                    break
        
        return result