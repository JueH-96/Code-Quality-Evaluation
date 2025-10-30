class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        min_len = float('inf')
        result = ""
        
        # Try all possible substrings
        for i in range(n):
            ones_count = 0
            for j in range(i, n):
                if s[j] == '1':
                    ones_count += 1
                
                # If we found a beautiful substring
                if ones_count == k:
                    current_len = j - i + 1
                    current_substring = s[i:j+1]
                    
                    # Update if this is shorter, or same length but lexicographically smaller
                    if (current_len < min_len or 
                        (current_len == min_len and current_substring < result)):
                        min_len = current_len
                        result = current_substring
                    break  # No need to extend further from this starting position
                
                # If we already have more than k ones, no point continuing
                elif ones_count > k:
                    break
        
        return result