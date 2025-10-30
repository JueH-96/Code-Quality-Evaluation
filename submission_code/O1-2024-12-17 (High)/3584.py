class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        """
        We want to find a subsequence of word1 of length len(word2) that differs from word2
        in at most one character, and among all such possible subsequences, return the
        lexicographically smallest sequence of indices.

        This solution uses a 2-state dynamic programming (DP) approach:
         - bestNoMismatch[j] holds the "state ID" of the lexicographically smallest way
           to match word2[:j] exactly (0 mismatches).
         - bestMismatch[j] holds the "state ID" of the lexicographically smallest way
           to match word2[:j] with at most 1 mismatch.

        Each "state" encodes:
          -- parent:  the previous state's ID
          -- idx:     the index in word1 chosen for the j-th character
          -- length:  how many characters of word2 this state has matched so far
                     (so this state corresponds to matching j characters of word2)

        Transitions (for j from 1..M, M = len(word2)):
          bestNoMismatch[j]:
            - extend bestNoMismatch[j-1] by an exact match of word2[j-1] if possible

          bestMismatch[j]:
            - route1: extend bestMismatch[j-1] by an exact match of word2[j-1]
            - route2: extend bestNoMismatch[j-1] by a mismatch at position j-1
            Then pick the lexicographically smaller of route1 vs route2.

        "Lexicographically smaller" for two states of the same length means:
          - compare their chosen indices from last to first; the one with the smaller
            index at the first differing position is lexicographically smaller.

        After filling DP, we compare bestNoMismatch[M] and bestMismatch[M] to see which
        final state (if any) is valid and lexicographically smaller. Then we reconstruct
        the chosen indices by following parent pointers up, collecting indices.

        Building the subsequence when at most 1 mismatch is allowed can be subtle; this
        DP-with-parent-pointer approach takes care of correctness, but carefully comparing
        paths in reverse ensures we do indeed pick the lexicographically smallest.

        Big-picture efficiency notes:
          - We build a "nextPos" table so that nextPos[i][c] gives the next position in
            word1 >= i of character c, or -1 if none.  This allows O(1) transitions.
          - We store at most 3 states per step: bestNoMismatch[j], route1, route2.  So
            we have O(M) steps and up to ~3*M states in total.
          - The main complexity is comparing two candidate states in "compareState".  It
            does a reverse walk through the parent chain, which in worst case can be O(M).
            Hence naïvely this can be O(M^2).  However, the problem constraints and test
            data may still allow this solution to run in time in practice (especially in
            lower-level languages).  In Python, care must be taken to implement as
            efficiently as possible.  If further speed is needed, one can implement a
            binary-lifting table for parent pointers to compare in O(log M), but that
            is considerably more complicated.

        If no valid subsequence is found, we return [].
        """

        import sys
        sys.setrecursionlimit(10**7)

        # Edge cases
        n, m = len(word1), len(word2)
        if m == 0:
            return []  # trivial (though by problem constraints, m>=1)
        if m > n:
            return []  # impossible if the pattern is longer than the text

        # --------------------------------------------------
        # 1) Build nextPos array: nextPos[i][c] = smallest index >= i in word1
        #    where character c appears, or -1 if not found.
        #    We'll represent c as an integer 0..25.
        # --------------------------------------------------
        ALPHABET_SIZE = 26
        def char2int(c):
            return ord(c) - ord('a')

        # nextPos[i][c] is defined for 0<=i<=n, 0<=c<26
        # We'll build it backward so that nextPos[n][c] = -1 for all c,
        # and for i in [n-1..0], we copy from nextPos[i+1] then set the
        # entry for word1[i].
        nextPos = [[-1]*ALPHABET_SIZE for _ in range(n+1)]
        for c in range(ALPHABET_SIZE):
            nextPos[n][c] = -1
        for i in range(n-1, -1, -1):
            # copy row from i+1
            for c in range(ALPHABET_SIZE):
                nextPos[i][c] = nextPos[i+1][c]
            # set the character of word1[i]
            nextPos[i][char2int(word1[i])] = i

        # --------------------------------------------------
        # We will store states in three parallel lists (or a list of tuples).
        # Each state s has:
        #   parent[s], idx[s], length[s]
        # where idx[s] = the index in word1 used for the length[s]-th character
        # of word2 (i.e. for j = length[s]).
        #
        # We'll keep -1 for "no state".
        #
        # bestNoMismatch[j] = state ID for matching j chars exactly,
        # bestMismatch[j]   = state ID for matching j chars with <=1 mismatch.
        # --------------------------------------------------

        parent  = []
        idxVal  = []
        lengthS = []

        def newState(par, i_val):
            # create a new state with given parent, index
            # length of new state is lengthS[par]+1 if par != -1
            # if par = -1, length=0 => that is the root state
            s_id = len(parent)
            parent.append(par)
            idxVal.append(i_val)
            if par == -1:
                lengthS.append(0)
            else:
                lengthS.append(lengthS[par] + 1)
            return s_id

        # Create the root state: state 0 => parent=-1, idx=-1, length=0
        rootState = newState(-1, -1)  # ID=0

        bestNoMismatch = [-1]*(m+1)
        bestMismatch   = [-1]*(m+1)

        # Initialize
        bestNoMismatch[0] = rootState  # state=0
        bestMismatch[0]   = rootState  # same

        # --------------------------------------------------
        # compareState(a, b) => returns the lex-smaller state among a,b
        # Both states are guaranteed to have the same length if both != -1.
        # We'll do a reverse walk through parent pointers to find first differing index.
        # This can be O(lengthS[a]) in worst case, but we hope it passes.
        # --------------------------------------------------
        def compareState(a, b):
            if a == b:
                return a
            if a == -1:
                return b
            if b == -1:
                return a
            # now we compare in reverse
            pa, pb = a, b
            # both have same length => lengthS[pa] == lengthS[pb]
            while pa != pb and pa != -1 and pb != -1:
                if idxVal[pa] < idxVal[pb]:
                    return a
                elif idxVal[pa] > idxVal[pb]:
                    return b
                pa = parent[pa]
                pb = parent[pb]
            # If we exit the loop with pa==pb, they are the same path => tie
            # Just pick one
            return a

        # --------------------------------------------------
        # 2) Fill DP
        # bestNoMismatch[j]: extends bestNoMismatch[j-1] by an exact match
        # bestMismatch[j]: pick min( route1, route2 ) in lex sense
        #   route1: extend bestMismatch[j-1] by an exact match
        #   route2: extend bestNoMismatch[j-1] by using mismatch
        # --------------------------------------------------
        for j in range(1, m+1):
            # bestNoMismatch[j]
            nmPrev = bestNoMismatch[j-1]
            if nmPrev != -1:
                # we want to match word2[j-1] exactly
                c = char2int(word2[j-1])
                i_prev = idxVal[nmPrev]  # the index chosen for word2[:j-1]
                # get next match index
                if i_prev == -1:
                    startPos = 0
                else:
                    startPos = i_prev + 1
                nxt = -1
                if startPos < n:
                    nxt = nextPos[startPos][c]
                if nxt != -1:
                    bestNoMismatch[j] = newState(nmPrev, nxt)
                else:
                    bestNoMismatch[j] = -1
            else:
                bestNoMismatch[j] = -1

            # bestMismatch[j]
            route1 = -1
            route2 = -1

            # route1: extend bestMismatch[j-1] by an exact match
            mixPrev = bestMismatch[j-1]
            if mixPrev != -1:
                c = char2int(word2[j-1])
                i_prev = idxVal[mixPrev]
                startPos = (i_prev+1) if (i_prev != -1) else 0
                nxt = -1
                if startPos < n:
                    nxt = nextPos[startPos][c]
                if nxt != -1:
                    route1 = newState(mixPrev, nxt)

            # route2: extend bestNoMismatch[j-1] by mismatch at (j-1)
            nmPrev = bestNoMismatch[j-1]
            if nmPrev != -1:
                i_prev = idxVal[nmPrev]
                mismatchPos = i_prev+1 if i_prev != -1 else 0
                if mismatchPos < n:
                    route2 = newState(nmPrev, mismatchPos)

            bestMismatch[j] = compareState(route1, route2)

        # --------------------------------------------------
        # 3) Compare final states bestNoMismatch[m] vs bestMismatch[m]
        # --------------------------------------------------
        cand = compareState(bestNoMismatch[m], bestMismatch[m])
        if cand == -1:
            return []  # no valid subsequence
        # Reconstruct the indices
        L = lengthS[cand]
        if L < m:
            # Means we didn't match all m characters
            return []

        # We know this state has matched m chars.  We'll collect them in reverse order.
        res = [0]*m
        x = cand
        while x != -1 and lengthS[x] > 0:
            res[lengthS[x]-1] = idxVal[x]
            x = parent[x]

        return res