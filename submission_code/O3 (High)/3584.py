from typing import List

class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n, m = len(word1), len(word2)

        # ----------  build suffix helper arrays  -------------
        # suff0[i]  : smallest index j s.t. word2[j:] can be matched
        #             EXACTLY inside word1 with indices >= i
        # suff1[i]  : smallest index j s.t. word2[j:] can be matched
        #             with AT MOST ONE mismatch inside word1 with indices >= i
        #
        # both arrays are of size n+1, where position n stands for empty suffix
        suff0 = [m]*(n+1)
        suff1 = [m]*(n+1)

        p0 = p1 = m          # pointers described above
        suff0[n] = p0
        suff1[n] = p1

        for i in range(n-1, -1, -1):
            c = word1[i]

            # --- update pointer for exact matching ---
            new_p0 = p0
            if p0 and c == word2[p0-1]:
                new_p0 = p0 - 1

            # --- update pointer for ≤1-mismatch matching ---
            new_p1 = p1                              # skip current char
            # case: mismatch already used earlier, we match current char exactly
            if p1 and c == word2[p1-1]:
                new_p1 = min(new_p1, p1-1)
            # case: we spend the (only) mismatch on current char
            if p0:                                   # we can align it to word2[p0-1]
                new_p1 = min(new_p1, p0-1)

            p0, p1 = new_p0, new_p1
            suff0[i] = p0
            suff1[i] = p1
        # ----------------------------------------------------

        ans: List[int] = []
        i = 0                    # current position we are looking at in word1
        mismatch_used = False    # have we already consumed the single mismatch?

        while len(ans) < m and i < n:
            needed = word2[len(ans)]          # character we need at this step
            c = word1[i]

            possible = False
            use_mismatch_now = False

            if c == needed:
                # character matches – check if the rest can be finished
                if mismatch_used:
                    if suff0[i+1] <= len(ans)+1:
                        possible = True
                else:   # mismatch still available for the tail
                    if suff1[i+1] <= len(ans)+1:
                        possible = True
            else:
                # characters differ – we may use the (only) mismatch here
                if not mismatch_used and suff0[i+1] <= len(ans)+1:
                    possible = True
                    use_mismatch_now = True

            if possible:                     # we take index i
                ans.append(i)
                if use_mismatch_now:
                    mismatch_used = True
                i += 1                       # next search starts right after chosen index
            else:
                i += 1                       # skip current position

        return ans if len(ans) == m else []