class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        max_length = 0
        forbidden_set = set(forbidden)
        last_forbidden_end = -1
        
        for i in range(len(word)):
            for j in range(max(last_forbidden_end + 1, i - 9), i + 1):
                if word[j:i+1] in forbidden_set:
                    last_forbidden_end = i
                    break
            max_length = max(max_length, i - last_forbidden_end)
        
        return max_length