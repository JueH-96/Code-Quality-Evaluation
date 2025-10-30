class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        max_len = 0
        right = len(word) - 1
        for left in range(len(word) - 1, -1, -1):
            for right in range(left + max_len, len(word)):
                if word[left:right+1] in forbidden_set:
                    break
                else:
                    max_len = max(max_len, right - left + 1)
        return max_len