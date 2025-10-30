class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        from collections import defaultdict

        forbidden_dict = defaultdict(set)
        for s in forbidden:
            l = len(s)
            forbidden_dict[l].add(s)
        
        max_length = 0
        left = 0
        n = len(word)
        
        for right in range(n):
            for l in forbidden_dict:
                if l > right - left + 1:
                    continue
                substr = word[right - l + 1 : right + 1]
                if substr in forbidden_dict[l]:
                    new_left = right - l + 2
                    if new_left > left:
                        left = new_left
            current_length = right - left + 1
            if current_length > max_length:
                max_length = current_length
        
        return max_length