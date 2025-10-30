class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        max_forbidden_len = max(len(s) for s in forbidden) if forbidden else 0
        left = 0
        max_len = 0
        n = len(word)
        
        for right in range(n):
            # Check all possible forbidden substrings ending at 'right'
            for length in range(1, min(max_forbidden_len, right + 1) + 1):
                start = right - length + 1
                current_sub = word[start : right + 1]
                if current_sub in forbidden_set:
                    left = max(left, start + 1)
            # Update the maximum length found
            current_length = right - left + 1
            if current_length > max_len:
                max_len = current_length
        
        return max_len