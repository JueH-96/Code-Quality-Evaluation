class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        forbidden_set = set(forbidden)
        n = len(word)
        max_length = 0
        i = 0

        while i < n:
            for j in range(i, min(i + 10, n)):
                if word[i:j+1] in forbidden_set:
                    break
            else:
                max_length = max(max_length, j - i + 1)
                i += 1
                continue
            i += 1

        return max_length