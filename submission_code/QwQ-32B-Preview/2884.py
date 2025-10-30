class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        n = len(word)
        forbidden_set = set(forbidden)
        
        # Precompute forbidden end positions
        forbidden_end_positions = [-1] * n
        for end in range(n):
            for len_f in range(1, 11):
                if end - len_f + 1 >= 0:
                    substring = word[end - len_f + 1 : end + 1]
                    if substring in forbidden_set:
                        forbidden_end_positions[end] = end
                        break
        
        # Sliding window to find the maximum valid substring length
        start = 0
        max_length = 0
        for end in range(n):
            if forbidden_end_positions[end] >= start:
                start = forbidden_end_positions[end] + 1
            max_length = max(max_length, end - start + 1)
        
        return max_length