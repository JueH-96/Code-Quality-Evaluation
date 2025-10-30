class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        n = len(word)
        k = numFriends
        # If there's only one friend, the only possible split is the whole word.
        if k == 1:
            return word

        # Precompute the offset for truncating substrings when i < k-1
        # For each start i, the longest valid substring ends at:
        #   j = i + (n - k + 1)   if i < k-1
        #   j = n                 otherwise
        N = n - k + 1

        best = ""
        for i in range(n):
            # compute the farthest end j for a valid segment starting at i
            if i < k - 1:
                j = i + N
            else:
                j = n
            # j > i always holds if k <= n
            cand = word[i:j]
            # keep the lexicographically largest candidate
            if cand > best:
                best = cand

        return best