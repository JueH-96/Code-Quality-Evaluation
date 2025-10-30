class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        max_len = 0
        start = 0
        
        while start < len(word):
            valid = True
            for length in range(1, len(word) - start + 1):
                substring = word[start:start + length]
                if any(substring.find(f) != -1 for f in forbidden):
                    break
                max_len = max(max_len, length)
            start += 1
        
        return max_len