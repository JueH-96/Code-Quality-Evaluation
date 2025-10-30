class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        if k == 1:
            return n
        if k >= n:
            return 1
        if n % k == 0:
            return n // k
        for i in range(1, n // k + 2):
            if word[:k * i] == word[k * i:2 * k * i]:
                return i