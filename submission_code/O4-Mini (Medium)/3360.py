from collections import Counter

class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        freq = Counter(word)
        # List of positive frequencies
        fvals = list(freq.values())
        if not fvals:
            return 0
        max_f = max(fvals)
        # If k large enough, no deletions needed
        if k >= max_f:
            return 0

        n = len(word)
        ans = n
        # Try each possible minimum frequency L for the remaining letters
        # and set H = min(max_f, L + k) as the allowed maximum frequency.
        for L in range(1, max_f + 1):
            H = L + k
            if H > max_f:
                H = max_f
            deletions = 0
            # Compute deletions needed to bring each letter's freq into [L, H]
            for f in fvals:
                if f < L:
                    deletions += f      # delete all occurrences
                elif f > H:
                    deletions += f - H  # cut down to H
            # Track the minimum deletions found
            if deletions < ans:
                ans = deletions
        return ans