from typing import List

class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n = len(word1)
        m = len(word2)
        # Precompute next occurrence table.
        # nxt[i][c] is the smallest index >= i where character c appears.
        # We have n+1 rows (with row n representing a dummy boundary)
        ALPH = 26
        nxt = [[-1] * ALPH for _ in range(n+1)]
        # For row n, no index is available.
        for c in range(ALPH):
            nxt[n][c] = -1
        
        # Build nxt from the end.
        for i in range(n-1, -1, -1):
            # First, copy row i+1
            # It is important to copy the list so that modification does not affect others.
            nxt[i] = nxt[i+1][:]
            # Current character:
            cind = ord(word1[i]) - 97
            nxt[i][cind] = i

        # Helper: from position 'start' in word1, try to match word2[j:] exactly,
        # returning the chosen indices or None if not possible.
        def get_seq(start: int, j: int) -> List[int]:
            res = []
            pos = start
            # For positions j ... m-1:
            for k in range(j, m):
                char = word2[k]
                cind = ord(char) - 97
                nxt_index = nxt[pos][cind]
                if nxt_index == -1:
                    return None
                res.append(nxt_index)
                pos = nxt_index + 1
                if pos > n:  # should not happen
                    break
            return res
        
        best_seq = None
        
        # Option 0: try an exact match (0 mismatches)
        seq0 = get_seq(0, 0)
        if seq0 is not None:
            best_seq = seq0
        
        # Option 1: try using exactly one mismatch at some position j_m in word2.
        # The idea:
        #   - Let prefix be the indices for positions 0 .. j_m-1 (which must be exactly matched).
        #   - At position j_m, we can choose the smallest possible index available.
        #   - Then match exactly for the remaining positions.
        for j_m in range(m):
            prefix = get_seq(0, j_m) if j_m > 0 else []
            if j_m > 0 and prefix is None:
                # if prefix matching fails, then any further j_m fail as well.
                break
            # determine next available position in word1 after the prefix
            # For an empty prefix, start at 0.
            pos = prefix[-1] + 1 if prefix else 0
            if pos >= n:  # no available index for mismatch
                continue
            # For the j_m-th character, we use a mismatch.
            # We choose the smallest index >= pos. That is simply pos.
            mismatch_index = pos
            
            # Now, try to match exactly the suffix word2[j_m+1:] starting from mismatch_index+1.
            suffix = get_seq(mismatch_index+1, j_m+1)
            if suffix is None:
                continue
            candidate_seq = prefix + [mismatch_index] + suffix
            # Compare lexicographically as lists of indices
            if best_seq is None or candidate_seq < best_seq:
                best_seq = candidate_seq
                
            # Because we are iterating j_m in increasing order, it might be that a later mismatch
            # yields a lexicographically larger sequence; however we must check all possible j_m.
        
        return best_seq if best_seq is not None else []