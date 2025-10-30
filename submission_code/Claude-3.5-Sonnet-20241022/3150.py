class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        min_len = float('inf')
        result = ""
        
        # Find all substrings with exactly k ones
        for i in range(n):
            ones = 0
            for j in range(i, n):
                if s[j] == '1':
                    ones += 1
                
                # If we found k ones
                if ones == k:
                    curr_len = j - i + 1
                    curr_str = s[i:j+1]
                    
                    # If current length is smaller than min_len found so far
                    if curr_len < min_len:
                        min_len = curr_len
                        result = curr_str
                    # If current length equals min_len, compare lexicographically
                    elif curr_len == min_len:
                        result = min(result, curr_str)
                    
                    break
                    
        return result