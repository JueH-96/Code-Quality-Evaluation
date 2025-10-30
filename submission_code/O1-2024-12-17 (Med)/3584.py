class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        """
        We want a subsequence of word1 (of length == len(word2)) whose characters
        can match word2 in all but at most one position (i.e. "almost equal").
        Among all such subsequences, return the lexicographically smallest sequence
        of indices.  If none exists, return [].

        ---------------------------------------------------------------
        OVERVIEW OF THE APPROACH:

        A direct greedy "take the earliest match, or mismatch if needed" can fail,
        because sometimes you should skip a smaller index now in order to save
        the single mismatch for a better place later.  Likewise, simply matching
        the whole word2 (if possible) does not necessarily yield the lexicographically
        smallest solution, because using an earlier mismatch may let you pick
        smaller indices for subsequent characters.

        However, we can solve it in O(n + m) time (where n=len(word1), m=len(word2))
        by a standard "next-position" table plus a 2-state dynamic programming
        that tracks (i) how far we've matched in word2 and (ii) whether we have
        used the mismatch yet.  Crucially, we also must keep track of the earliest
        position in word2 at which we used the mismatch (so we can break ties
        lexicographically).

        ---------------------------------------------------------------
        STEP 1: Build the "nextPos" table for word1.

            nextPos[pos][c] = the smallest index j >= pos in word1
                              such that word1[j] == c
                              or -1 if none exists.

            We will do this by going from right to left in word1:
              - Initialize nextPos[n][c] = -1 for all c in [0..25].
              - For i in [n-1 .. 0], copy nextPos[i+1], then set
                nextPos[i][ word1[i] - 'a' ] = i.
            
            Then, if you are currently at index x in word1 (meaning you must pick
            the next character after x), and you want to match character ch,
            you can jump directly to j = nextPos[x+1][ ch ]; if j == -1,
            then there is no further match for ch.

        ---------------------------------------------------------------
        STEP 2: Define DP states:

            We will have dp0[i] = the smallest index in word1 at which we place the
                i-th character of word2, assuming ZERO mismatches used so far.
                If dp0[i] = -1, it means we cannot form word2[:i+1] with 0 mismatches.
            
            We will have dp1[i] = the smallest index in word1 at which we place the
                i-th character of word2, allowing AT MOST ONE mismatch in total.
                If dp1[i] = -1, it means we cannot form word2[:i+1] with <=1 mismatch.

            Additionally, for dp1 we need to track which position in word2 we used
            the mismatch (if we did).  Let mismatchPos1[i] = the index in word2
            at which the mismatch was used on the way to dp1[i], or -1 if no mismatch
            was used yet.

            Why do we need mismatchPos1[i]?  Because when we combine two candidates
            for dp1[i], we must pick the one that yields the lexicographically
            smaller final subsequence.  The first difference between two subsequences
            will appear at the earliest mismatch in word2.  So we compare mismatchPos
            first (the smaller mismatchPos is lexicographically better), and if those
            are the same, we compare the final picked index if needed.

        ---------------------------------------------------------------
        STEP 3: Recurrence / transitions.

            Let w2[i] = word2[i],  c = (ord(w2[i]) - ord('a')).

            For dp0[i]: (0 mismatches used)
                - If i == 0:
                    dp0[0] = nextPos[0][  w2[0]-'a' ]
                    (the earliest match from the start)
                  If dp0[0] == -1, no solution with zero mismatches for the first char.
                - If i > 0 and dp0[i-1] != -1:
                    dp0[i] = nextPos[ dp0[i-1] + 1 ][  c ]
                  else dp0[i] = -1

            For dp1[i]: (<=1 mismatch allowed in total)
                We have two ways to form word2[:i+1] with <=1 mismatch:

                (A) Extend dp1[i-1] by matching the i-th character exactly:
                     if dp1[i-1] != -1:
                        j = nextPos[ dp1[i-1] + 1 ][ c ]
                        if j != -1, then dp1[i] can be j
                        mismatchPos1[i] = mismatchPos1[i-1]  (reuse the same mismatch info)
                
                (B) If we haven't used a mismatch yet, we can mismatch the i-th char:
                     that means we come from dp0[i-1] (so 0 mismatches used for the
                     previous i characters), pick j = dp0[i-1] + 1, if j < len(word1).
                     We do not care what word1[j] is, because we will use the mismatch
                     here to pretend it matches w2[i].  Then mismatchPos1[i] = i
                     (the mismatch is used exactly for word2[i]).

                We combine these two possibilities; if both are valid, we pick the one
                that yields the lexicographically smaller final subsequence.  By the
                problem statement, that boils down to:
                   1) comparing mismatchPos (the earliest i in word2 where mismatch used);
                   2) if tie, pick the smaller chosen index j.

        ---------------------------------------------------------------
        STEP 4: Reconstructing the actual subsequence indices.

            We also need to store parent pointers so we can walk back and see which index
            we used for each character.

            Let parent0[i] = the index i-1 from which dp0[i] was formed, i.e. i-1,
            and we also store chosenIndex0[i] = dp0[i].
            Similarly for dp1, we store parent1[i] (which is either i-1 in state dp1, or i-1 in state dp0 if we used mismatch),
            along with chosenIndex1[i], and mismatchPos1[i].

            Once we finish i = m-1 (where m = len(word2)), we will look at:
               - If dp0[m-1] != -1, that is a valid 0-mismatch solution.  Then its mismatchPos = -1,
                 which is lexicographically better than any real mismatch.  We can immediately take it.
               - Else if dp1[m-1] != -1, we use dp1.  That at least is a 1-mismatch solution.
               - Otherwise return [].

            We then trace back from i = m-1 in the chosen state (dp0 or dp1) to build
            the subsequence of indices in reverse, and finally reverse it.

        ---------------------------------------------------------------
        STEP 5: Complexity

            - Building nextPos takes O(26 * n) which is feasible (n up to 3e5).
            - Filling dp0, dp1 arrays each of length m (up to 3e5) is O(m).
              Each step is O(1) since it is just a couple of table lookups.
            - Storing/reconstructing parent pointers is also O(m).
            Overall this is O(n + m), which is acceptable for n, m up to 3*10^5.

        ---------------------------------------------------------------
        EXAMPLES:

        Example 1:
          word1 = "vbcca", word2 = "abc"
          The dp approach will find that the 0-mismatch solution is impossible,
          but a 1-mismatch solution can pick indices [0,1,2], using mismatch at i=0
          (where 'v' is changed to 'a').  That yields the lexicographically smallest
          solution ([0,1,3] is also valid but is larger in the last index).

        Example 2:
          word1 = "bacdc", word2 = "abc"
          The dp approach finds that you can match 'a' at index1 (no mismatch), then
          mismatch 'b' at index2, then match 'c' at index4.  That yields [1,2,4].

        Example 3:
          word1 = "aaaaaa", word2 = "aaabc"
          We can never form 'b' and 'c' in order with at most one mismatch, so dp1
          will end up -1 at some step, returning [].

        Example 4:
          word1 = "abc", word2 = "ab"
          We can match exactly with dp0 => indices [0,1].

        ---------------------------------------------------------------
        Implementation details below.
        """

        import sys
        sys.setrecursionlimit(10**7)
        from string import ascii_lowercase

        n = len(word1)
        m = len(word2)
        if m > n:
            return []

        # -----------------------------------------------------------
        # Build nextPos array: nextPos[i][c] = smallest j >= i s.t. word1[j] == c
        # -----------------------------------------------------------
        # We'll map characters 'a'..'z' to 0..25
        ALPH = 26
        nextPos = [[-1]*ALPH for _ in range(n+1)]
        # Initialize base case
        for c in range(ALPH):
            nextPos[n][c] = -1
        
        # Fill from right to left
        for i in range(n-1, -1, -1):
            # copy from row i+1
            for c in range(ALPH):
                nextPos[i][c] = nextPos[i+1][c]
            # update for the character at word1[i]
            cindex = ord(word1[i]) - ord('a')
            nextPos[i][cindex] = i

        # -----------------------------------------------------------
        # DP arrays:
        #   dp0[i] = index in word1 where we placed word2[i] with 0 mismatches
        #   parent0[i] = i-1
        #   chosen0[i] = the actual index used
        #
        #   dp1[i] = index in word1 where we placed word2[i] with <=1 mismatch
        #   mismatchPos1[i] = the earliest i' in [0..i] at which we used the mismatch
        #       or -1 if we haven't used it yet
        #   parent1[i], chosen1[i] similarly to track reconstruction
        #
        # We'll store -1 if something is impossible.
        # -----------------------------------------------------------

        dp0 = [-1]*m
        parent0 = [-1]*m   # always i-1, but we keep for consistency
        chosen0 = [-1]*m   # which index in word1 is used

        dp1 = [-1]*m
        mismatchPos1 = [-1]*m
        parent1 = [-1]*m
        chosen1 = [-1]*m

        # ---- Initialize dp0[0] and dp1[0] ----
        first_ch = ord(word2[0]) - ord('a')
        # dp0[0]: must match word2[0] exactly from the start of word1
        dp0[0] = nextPos[0][ first_ch ]
        chosen0[0] = dp0[0]

        # dp1[0]: we can either match or mismatch
        #   - match => j = dp0[0]
        #   - mismatch => j = 0, provided 0 < n
        # We pick whichever yields the lexicographically smaller subsequence.
        # Mismatch would mean mismatchPos=0, last index = 0
        # Match would mean mismatchPos=-1, last index = dp0[0]
        # We compare mismatchPos first: -1 < 0, so if dp0[0] != -1, that means
        #  not using mismatch is lexicographically better (earlier mismatchPos).
        # But wait, "no mismatch used" => mismatchPos=-1 is considered 'better'
        # than mismatchPos=0 in strict lex ordering of the subsequence?
        # Actually, the problem statement says we want the lexicographically smallest
        # array of indices.  The "earliest mismatch in word2" is how we break ties
        # between distinct subsequences of the same length.  But here the sequences
        # might differ from their first element if dp0[0] is bigger than 0.
        #
        # Concretely:
        #   If dp0[0] = -1, we can only mismatch => dp1[0] = 0 (if n>0)
        #   If dp0[0] != -1, let match_index = dp0[0].
        #      mismatch_index = 0.
        #   Compare those two subsequences of length 1: [match_index] vs [0].
        #   The smaller array in normal numeric order is whichever has the smaller
        #   single integer.  So if match_index > 0, mismatch yields a smaller index.
        #   But in that case, mismatchPos=0 is used earlier in word2.  By the problem's
        #   statement, the subsequence with the smaller index at position 0 is
        #   lexicographically smaller.  So we prefer mismatch if 0 < match_index.
        #
        # Summarize:
        #   If dp0[0] == -1 => dp1[0] = 0 if 0<n, mismatchPos=0
        #   else we have two candidates:
        #     candidate A: mismatch => index=0, mismatchPos=0
        #     candidate B: match => index=dp0[0], mismatchPos=-1
        #   We pick the array [0] if 0 < dp0[0], or [dp0[0]] if dp0[0] <= 0.
        #   But dp0[0] can't be < 0 except if -1 => not found.  If dp0[0]==0, that
        #   means word1[0] actually matches word2[0], so [0] using no mismatch is
        #   the same final array.  Then mismatchPos is -1 which is "earlier" in
        #   mismatchPos ordering, so we pick that.  In effect, if dp0[0] == 0, we prefer
        #   not to use mismatch.  If dp0[0] > 0, we prefer to mismatch at 0.  If dp0[0] = -1,
        #   we must mismatch if we want anything at all.
        #

        match_index_0 = dp0[0]  # could be -1 if no match
        # mismatch always picks index=0 if 0 < n
        mismatch_index_0 = 0 if n>0 else -1

        if match_index_0 == -1 and mismatch_index_0 == -1:
            # then dp1[0] = -1 => no way
            dp1[0] = -1
            mismatchPos1[0] = -1
        elif match_index_0 == -1:
            # must mismatch
            dp1[0] = mismatch_index_0
            mismatchPos1[0] = 0
            chosen1[0] = mismatch_index_0
            parent1[0] = -1
        else:
            # we have two possible picks
            # candidate A: mismatch => index=0, mismatchPos=0
            # candidate B: match => index=match_index_0, mismatchPos=-1
            # prefer the one that yields smaller first index in the final subsequence
            candA = mismatch_index_0
            candB = match_index_0
            if candA < candB:
                # mismatch yields smaller final index
                dp1[0] = candA
                mismatchPos1[0] = 0
                chosen1[0] = candA
                parent1[0] = -1
            elif candA > candB:
                # matching yields smaller
                dp1[0] = candB
                mismatchPos1[0] = -1
                chosen1[0] = candB
                parent1[0] = -1
            else:
                # candA == candB => then mismatchPos is -1 vs 0 => -1 is lexicographically "better"
                # because that means no mismatch used yet
                dp1[0] = candB
                mismatchPos1[0] = -1
                chosen1[0] = candB
                parent1[0] = -1
        
        # fill chosen0[0], parent0[0]
        parent0[0] = -1

        # -----------------------------------------------------------
        # Fill dp0[i], dp1[i] for i=1..m-1
        # -----------------------------------------------------------
        for i in range(1, m):
            c = ord(word2[i]) - ord('a')

            # dp0[i]:
            if dp0[i-1] == -1:
                dp0[i] = -1
            else:
                nxt = -1
                startPos = dp0[i-1] + 1
                if startPos <= n:
                    nxt = nextPos[startPos][c]
                dp0[i] = nxt
            chosen0[i] = dp0[i]
            parent0[i] = i-1

            # dp1[i]: gather two candidates
            # Candidate 1 (extend dp1[i-1] by a match):
            cand1_idx = -1
            cand1_mpos = -1
            cand1_par = -1
            if dp1[i-1] != -1:
                startPos = dp1[i-1] + 1
                idx = -1
                if startPos <= n:
                    idx = nextPos[startPos][c]
                if idx != -1:
                    cand1_idx = idx
                    cand1_mpos = mismatchPos1[i-1]
                    cand1_par = i-1

            # Candidate 2 (use mismatch now, coming from dp0[i-1]):
            cand2_idx = -1
            cand2_mpos = -1
            cand2_par = -1
            if dp0[i-1] != -1:
                # we can mismatch i at dp0[i-1]+1
                j = dp0[i-1] + 1
                if j < n:  # we at least have an index to place it
                    cand2_idx = j
                    cand2_mpos = i  # mismatch used for word2[i]
                    cand2_par = i-1

            # Now pick the better of cand1 vs cand2
            # Compare by (mismatchPos, index)
            # The smaller mismatchPos is better; if they tie, the smaller index is better.
            def better(mposA, idxA, mposB, idxB):
                """
                Return True if (mposA, idxA) is lexicographically smaller than (mposB, idxB),
                given that -1 < any real i for mismatchPos.  (Because mismatchPos==-1 means
                'no mismatch used yet', which is effectively "better" than any real mismatch).
                """
                if mposA == -1 and mposB != -1:
                    return True  # no mismatch used is strictly better
                if mposB == -1 and mposA != -1:
                    return False
                # now either both -1 or both >= 0
                if mposA < mposB:
                    return True
                if mposA > mposB:
                    return False
                # tie in mismatchPos
                return idxA < idxB

            # pick among cand1 and cand2
            # We'll define dp1[i], mismatchPos1[i], chosen1[i], parent1[i]
            best_idx = -1
            best_mpos = 10**10  # large sentinel
            best_par = -1

            # function to try a candidate
            def try_cand(ci, cm, cp, best_i, best_m, best_p):
                if ci == -1:
                    return (best_i, best_m, best_p)  # invalid
                # compare with current best
                if best_i == -1:
                    # no existing best => take this
                    return (ci, cm, cp)
                else:
                    # compare
                    if better(cm, ci, best_m, best_i):
                        return (ci, cm, cp)
                    else:
                        return (best_i, best_m, best_p)

            (best_idx, best_mpos, best_par) = try_cand(cand1_idx, cand1_mpos, cand1_par,
                                                       best_idx, best_mpos, best_par)
            (best_idx, best_mpos, best_par) = try_cand(cand2_idx, cand2_mpos, cand2_par,
                                                       best_idx, best_mpos, best_par)

            dp1[i] = best_idx
            if best_idx == -1:
                mismatchPos1[i] = -1
                chosen1[i] = -1
                parent1[i] = -1
            else:
                mismatchPos1[i] = best_mpos if best_mpos != 10**10 else -1
                chosen1[i] = best_idx
                parent1[i] = best_par

        # -----------------------------------------------------------
        # Now decide which final solution to use: dp0[m-1] or dp1[m-1].
        # If dp0[m-1] != -1, that means we formed a perfect match (0 mismatch).
        # That is effectively mismatchPos=-1, which is lexicographically the earliest
        # mismatch position possible.  Hence that is strictly better than any dp1 solution
        # that uses mismatch at some real index.  If dp1 never used mismatch, it might
        # also have mismatchPos=-1, in which case we need to compare the final indices.
        #
        # But in fact if dp0[m-1] != -1, that subsequence is EXACT.  Then dp1's sequence
        # for the same mismatchPos=-1 would pick the very same indices, so they'd tie.
        # We can just prefer dp0 in that case.  If dp0[m-1] == -1, we look at dp1[m-1].
        # -----------------------------------------------------------

        if dp0[m-1] == -1 and dp1[m-1] == -1:
            return []

        use_dp0 = False
        if dp0[m-1] != -1:
            # then we have a valid 0-mismatch solution => prefer it
            use_dp0 = True
        else:
            # must use dp1 if dp1[m-1] != -1
            if dp1[m-1] == -1:
                return []
            else:
                use_dp0 = False

        # -----------------------------------------------------------
        # Reconstruct the subsequence of indices
        # -----------------------------------------------------------
        res = [0]*m
        if use_dp0:
            # Walk backwards from i=m-1 down to 0
            i = m-1
            cur_state = 0  # meaning dp0
            while i >= 0:
                res[i] = chosen0[i]
                i = parent0[i]
            # now res holds them in proper order
        else:
            # use dp1
            i = m-1
            cur_state = 1
            while i >= 0:
                res[i] = chosen1[i]
                i = parent1[i]

        # If any of them is -1, that would be invalid, but we already checked.
        return res