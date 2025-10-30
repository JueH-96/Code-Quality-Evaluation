class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        max_forbidden = max(len(f) for f in forbidden)
        forbidden_set = set(forbidden)
        left = 0
        max_length = 0
        
        for right in range(len(word)):
            start = max(left, right - max_forbidden + 1)
            new_left = left
            for j in range(start, right + 1):
                substr = word[j:right+1]
                if substr in forbidden_set:
                    new_left = max(new_left, j + 1)
            left = new_left
            current_length = right - left + 1
            if current_length > max_length:
                max_length = current_length
        
        return max_length