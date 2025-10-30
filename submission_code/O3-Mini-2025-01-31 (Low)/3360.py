class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        # Count frequency of each character
        from collections import Counter
        freq = Counter(word)
        # We only have at most 26 letters so we work with these frequencies.
        frequencies = list(freq.values())
        n = len(word)
        if not frequencies:
            return 0  # empty string
        
        # The idea: We want to keep as many characters as possible
        # while adjusting (deleting some occurrences) so that the resulting
        # set of kept letters (if a letter is kept at all, then it gets some positive frequency)
        # meets: for any two letters with positive frequency, the absolute difference is <= k.
        # Let m be the minimum frequency among letters we keep.
        # Then every kept letter’s frequency must lie in the interval [m, m+k].
        # For each letter with original frequency f:
        #   - If f < m, we cannot include that letter (delete all of them).
        #   - Otherwise, we can keep at most min(f, m+k) occurrences from that letter.
        # We try all possible values of m (from 1 to max frequency) and choose the best.
        max_freq = max(frequencies)
        best_keep = 0
        # m represents the intended minimum frequency for any letter we keep.
        for m in range(1, max_freq+1):
            current_keep = 0
            for f in frequencies:
                if f < m:
                    # Cannot meet the requirement: we opt to delete all occurrences of this letter.
                    continue
                else:
                    # We can keep at most m+k occurrences (we want to remain in range [m, m+k])
                    current_keep += min(f, m+k)
            best_keep = max(best_keep, current_keep)
            
        # The answer is total deletions required = total letters - maximum letters we can keep.
        return n - best_keep