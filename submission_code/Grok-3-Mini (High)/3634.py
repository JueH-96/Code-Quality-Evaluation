import bisect
from collections import defaultdict

class Solution:
    def calculateScore(self, s: str) -> int:
        # Create mirror map
        mirror_map = {}
        for i in range(26):
            char = chr(ord('a') + i)
            mirror_char = chr(ord('a') + 25 - i)
            mirror_map[char] = mirror_char
        
        # Create indices for each character
        char_indices = defaultdict(list)
        for idx in range(len(s)):
            char_indices[s[idx]].append(idx)  # Lists are sorted since appended in order
        
        score = 0
        n = len(s)
        
        for i in range(n):
            curr_char = s[i]
            mirror_c = mirror_map[curr_char]
            mirror_idx_list = char_indices[mirror_c]
            
            # Binary search to find the leftmost position where value >= i
            pos = bisect.bisect_left(mirror_idx_list, i)
            
            if pos > 0:
                j = mirror_idx_list[pos - 1]
                if j < i:
                    # Pair them, add to score
                    score += i - j
                    
                    # Remove j from mirror_idx_list
                    del mirror_idx_list[pos - 1]
                    
                    # Remove i from its own char index list
                    curr_idx_list = char_indices[curr_char]
                    idx_pos = bisect.bisect_left(curr_idx_list, i)
                    del curr_idx_list[idx_pos]
        
        return score