class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        from typing import List

        # Attempt to find a subsequence of word1 that aligns with word2
        # using at most 'max_mismatch' character mismatches.
        # Returns the lexicographically smallest sequence of indices if successful,
        # or None if not possible.
        def find_subsequence_with_mismatches(word1, word2, max_mismatch):
            i, j = 0, 0
            mismatch_used = 0
            chosen_indices = []
            n1, n2 = len(word1), len(word2)

            while i < n1 and j < n2:
                if word1[i] == word2[j]:
                    # Perfect match, take it
                    chosen_indices.append(i)
                    i += 1
                    j += 1
                else:
                    # Mismatch case
                    if mismatch_used < max_mismatch:
                        mismatch_used += 1
                        chosen_indices.append(i)
                        i += 1
                        j += 1
                    else:
                        i += 1

            # If we've matched all of word2, return the chosen indices
            if j == n2:
                return chosen_indices
            return None

        # First try with 0 mismatches (exact subsequence match)
        seq0 = find_subsequence_with_mismatches(word1, word2, 0)
        if seq0 is not None:
            return seq0

        # Otherwise, try with 1 mismatch allowed
        seq1 = find_subsequence_with_mismatches(word1, word2, 1)
        if seq1 is not None:
            return seq1

        # If neither works, return an empty list
        return []