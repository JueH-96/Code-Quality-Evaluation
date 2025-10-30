class Solution:
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        t = 1
        while True:
            if t * k <= n:
                # forced portion length is n - t*k.
                forced_len = n - t * k
                # Check if the forced portion of the final string would match the beginning of word.
                if word[t * k:] == word[:forced_len]:
                    return t
            else:
                # t*k > n: then the forced portion length is negative,
                # so we have complete freedom to choose the resulting string.
                return t
            t += 1

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.minimumTimeToInitialState("abacaba", 3))   # Expected output: 2
    print(sol.minimumTimeToInitialState("abacaba", 4))   # Expected output: 1
    print(sol.minimumTimeToInitialState("abcbabcd", 2))  # Expected output: 4