class Solution:
    def maximumLength(self, s: str) -> int:
        """
        Longest length L such that there exists a single character c for which a
        special substring (c repeated L times) appears at least three times in s.
        """
        n = len(s)
        # Check lengths from large to small, return the first that is possible.
        for L in range(n, 0, -1):
            for ch in 'abcdefghijklmnopqrstuvwxyz':
                occ = 0          # how many substrings ch*L we have seen
                i = 0
                while i < n:
                    if s[i] != ch:
                        i += 1
                        continue
                    # we are at the start of a run of `ch`
                    j = i
                    while j < n and s[j] == ch:
                        j += 1
                    run_len = j - i
                    if run_len >= L:
                        # this run contributes (run_len - L + 1) substrings of length L
                        occ += run_len - L + 1
                    if occ >= 3:          # early exit – we already have enough
                        return L
                    i = j                 # jump to the first char after the run
        return -1