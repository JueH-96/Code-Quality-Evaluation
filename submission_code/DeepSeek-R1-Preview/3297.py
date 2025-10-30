class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        for t in range(1, n + 1):
            p = (t * k) % n
            if p == 0:
                return t
            if word.startswith(word[p:]):
                return t
        return n  # This line is theoretically unreachable as per problem constraints