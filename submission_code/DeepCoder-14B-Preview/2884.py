from collections import defaultdict

class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_dict = defaultdict(set)
        max_len = 0
        for s in forbidden:
            l = len(s)
            forbidden_dict[l].add(s)
            if l > max_len:
                max_len = l
        
        left = 0
        max_length = 0
        
        for right in range(len(word)):
            # Check all possible forbidden substrings ending at 'right'
            for l in forbidden_dict:
                if l > right + 1:
                    continue
                start = right - l + 1
                if start < 0:
                    continue
                if start < left:
                    continue
                substring = word[start:start+l]
                if substring in forbidden_dict[l]:
                    left = max(left, start + l)
            
            current_length = right - left + 1
            if current_length > max_length:
                max_length = current_length
        
        return max_length