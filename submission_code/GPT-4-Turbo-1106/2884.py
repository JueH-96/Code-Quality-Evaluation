class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        max_length = 0
        start = 0
        
        while start < len(word):
            valid = True
            for length in range(1, len(word) - start + 1):
                if any(word[start:start+length].find(f) != -1 for f in forbidden):
                    valid = False
                    break
                max_length = max(max_length, length)
            if not valid:
                start += 1
            else:
                break
        
        return max_length