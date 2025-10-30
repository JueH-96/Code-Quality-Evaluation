class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        max_len = 0
        n = len(word)
        for i in range(n):
            for j in range(i, n):
                substring = word[i:j+1]
                is_valid = True
                for f in forbidden_set:
                    if f in substring:
                        is_valid = False
                        break
                if is_valid:
                    max_len = max(max_len, len(substring))
        return max_len