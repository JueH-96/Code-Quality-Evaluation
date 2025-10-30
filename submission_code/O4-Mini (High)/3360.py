class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        from collections import Counter

        # Count frequencies of each character
        freq = Counter(word)
        counts = list(freq.values())
        counts.sort()
        
        total = len(word)
        max_count = counts[-1]
        max_kept = 0
        
        # Try every possible minimum frequency L in the final string
        for L in range(1, max_count + 1):
            R = L + k
            kept = 0
            # For each character, if its original count c >= L, we can keep
            # min(c, R) of them; otherwise we must delete them all.
            for c in counts:
                if c < L:
                    continue
                if c > R:
                    kept += R
                else:
                    kept += c
            max_kept = max(max_kept, kept)
        
        # Minimum deletions = original length - maximum kept
        return total - max_kept