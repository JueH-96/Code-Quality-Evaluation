class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        for t in range(1, n // k + 1):
            if n % k == 0 or t * k < n:
                valid = True
                for i in range(t * k, n):
                    if word[i] != word[i - t * k]:
                        valid = False
                        break
                if valid:
                    return t
        return n // k + (n % k > 0)