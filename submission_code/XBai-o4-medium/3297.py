class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        t_max = (n + k - 1) // k
        for t in range(1, t_max + 1):
            s = t * k
            if s >= n:
                return t
            else:
                prefix_length = n - s
                if word[:prefix_length] == word[s:s + prefix_length]:
                    return t
        return t_max