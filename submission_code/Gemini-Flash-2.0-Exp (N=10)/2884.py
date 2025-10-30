class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        max_len = 0
        start = 0
        for end in range(len(word)):
            for i in range(start, end + 1):
                if word[i:end+1] in forbidden_set:
                    start = end + 1
                    break
            max_len = max(max_len, end - start + 1)
        return max_len