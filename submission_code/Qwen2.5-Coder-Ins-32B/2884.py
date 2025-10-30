class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        n = len(word)
        max_length = 0
        
        for start in range(n):
            current_substring = ""
            for end in range(start, n):
                current_substring += word[end]
                if current_substring in forbidden_set:
                    break
                max_length = max(max_length, end - start + 1)
        
        return max_length