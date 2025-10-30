class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        # We want to gather all possible contiguous segments that can occur 
        # when splitting the entire string 'word' into exactly 'numFriends' 
        # non-empty parts. Then, from all those possible segments, we want 
        # the lexicographically largest one.
        #
        # Observing the problem, a segment is valid if it can appear as one 
        # contiguous part in some valid partition of the string into numFriends parts.
        #
        # A substring word[i..j] (0-based indexing) is a valid segment if 
        # and only if the remaining characters (i on the left and (n-1-j) 
        # on the right) can be split into (numFriends-1) non-empty parts. 
        #
        # Concretely, there must exist k in [0..(numFriends-1)] such that:
        #   - The left side (length = i) can form k parts => i >= k
        #   - The right side (length = n-1-j) can form (numFriends-1-k) parts => (n-1-j) >= (numFriends-1-k)
        #
        # This is equivalent to requiring an integer k in the intersection
        #   0 <= k <= i
        #   and
        #   0 <= (numFriends - 1 - k) <= (n - 1 - j)
        #
        # After algebra, one can derive a simpler check for i, j:
        #   max(0, j - (n - numFriends)) <= min(i, numFriends - 1).
        #
        # However, explicitly enumerating all substrings can be O(n^2), 
        # and comparing them lexicographically can push us toward O(n^3), 
        # which is too large for n=5000.
        #
        # Instead, we note that among all valid substrings, the lexicographically
        # largest must be a prefix of some suffix of 'word'. Indeed, any substring 
        # is a prefix of the suffix that starts at the same position.
        #
        # Hence, the strategy:
        #   1) Build a suffix array for 'word' (in O(n log n) time).
        #   2) Sort suffixes in descending lexicographical order.
        #   3) For each suffix index i in descending lex order, 
        #      figure out the range of substring lengths L that yield a valid segment:
        #         LB = max(1, (n - numFriends) - i + 1)
        #         UB = (depends on whether i <= numFriends-1 or not)
        #         Specifically:
        #            if i <= numFriends-1:
        #               UB = min(n - i, n - numFriends + 1)
        #            else:
        #               UB = n - i
        #         We require LB <= UB for feasibility.
        #      If feasible, pick L = UB (the longest valid prefix of that suffix).
        #      Return word[i : i + L] immediately, because the suffixes are 
        #      iterated in descending order, so the first feasible one is 
        #      the global lexicographically largest.
        #
        # Let's implement this plan.

        s = word
        n = len(s)

        # -------------------------
        # 1) Build suffix array (O(n log n)) 
        # -------------------------
        # We'll implement a standard prefix-doubling approach.

        # K: current rank array, used to sort suffixes by first 2^step characters
        # SA: the suffix array (array of indices)
        SA = list(range(n))
        K  = [ord(c) for c in s] + [-1]  # rank array extended with sentinel -1
        tmp = [0]*n

        step = 1
        # We'll sort by 2^h length substrings at each iteration
        def compare(i, j):
            # Compare based on rank K[i], K[i+2^(step-1)], etc.
            if K[i] != K[j]:
                return K[i] - K[j]
            else:
                ri = K[i+step] if i+step < n else -1
                rj = K[j+step] if j+step < n else -1
                return ri - rj

        step = 1
        while step <= n:
            # Sort SA by (rank[i], rank[i+step]) pairs
            SA.sort(key=lambda x: (K[x], K[x+step] if x+step < n else -1))
            # Compute new ranks in tmp
            tmp[SA[0]] = 0
            for iSA in range(1, n):
                tmp[SA[iSA]] = tmp[SA[iSA-1]] + (1 if compare(SA[iSA-1], SA[iSA])<0 else 0)
            for iSA in range(n):
                K[SA[iSA]] = tmp[SA[iSA]]
            step <<= 1
            if K[SA[-1]] == n-1:  # all ranks distinct => done
                break

        # -------------------------
        # 2) Iterate suffixes in descending lex order 
        #    i.e. we want to check suffixes in the order of descending rank.
        #    But we have built an ascending order suffix array. So let's invert it.
        # -------------------------
        # We also have K array storing final ranks. The suffix with largest rank 
        # is at K = n - 1, the second largest rank has K = n - 2, etc.
        # Alternatively, we can just iterate SA in reverse because SA is in 
        # ascending lex order, so reversed(SA) is in descending lex order.

        # We'll define a function to check feasibility given i (start index).
        def get_longest_prefix_length(i):
            # LB = max(1, (n - numFriends) - i + 1)
            LB = max(1, (n - numFriends) - i + 1)
            # If i <= numFriends - 1 => UB = min(n - i, n - numFriends + 1)
            # else => UB = n - i
            if i <= numFriends - 1:
                UB = min(n - i, n - numFriends + 1)
            else:
                UB = n - i
            if LB > UB:
                return 0  # no feasible length
            return UB  # pick the largest feasible length

        # 3) Check suffixes from largest to smallest in lex ordering
        for i in reversed(SA):
            length = get_longest_prefix_length(i)
            if length > 0:
                return s[i : i + length]

        # If somehow none is found (should not happen), return empty string
        return ""