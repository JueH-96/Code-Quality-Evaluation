class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        # Count frequencies of each character
        from collections import Counter
        cnt = Counter(word)
        freqs = list(cnt.values())
        n = len(word)
        if not freqs:
            return 0
        
        max_cnt = max(freqs)
        best_kept = 0
        
        # Try every possible minimum frequency L from 1 to max_cnt
        # Then the maximum allowed frequency is R = L + k
        for L in range(1, max_cnt + 1):
            R = L + k
            kept = 0
            # For each character frequency c:
            # - If c < L, we drop all occurrences of that character.
            # - Otherwise we keep up to min(c, R).
            for c in freqs:
                if c >= L:
                    kept += min(c, R)
            # Track the maximum total we can keep
            if kept > best_kept:
                best_kept = kept
        
        # Minimum deletions = total length - best we can keep
        return n - best_kept