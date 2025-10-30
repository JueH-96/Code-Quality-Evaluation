def main():
    import sys
    sys.setrecursionlimit(10**7)

    MOD = 998244353

    input_data = sys.stdin.read().strip().split()
    N, K = map(int, input_data[:2])
    S = input_data[2]

    #
    # -----------------------------
    # Idea and Explanation
    # -----------------------------
    #
    # Recall the problem’s definition of a “good” substring T (only A,B,C) as one
    # that can be completely removed by repeatedly:
    #   - removing two identical letters, or
    #   - removing one A, one B, one C at once.
    #
    # A well-known necessary-and-sufficient condition for such a string T to be “good”:
    #   Let xA, xB, xC be the counts of A,B,C in T.
    #   Then T is “good” if and only if EITHER
    #       (1) all of xA, xB, xC are even,
    #     OR (2) all of xA, xB, xC are odd and all ≥ 1.
    #
    # In terms of parity, condition (1) is “(xA % 2, xB % 2, xC % 2) = (0,0,0)” 
    # and condition (2) is “(1,1,1)” PLUS “xA,xB,xC ≥ 1”.
    #
    # However, checking each substring’s counts after replacing '?' is complicated
    # if we attempt to brute force all 3^(# of '?') assignments.
    #
    # -----------------------------
    # Parity-based Reformulation
    # -----------------------------
    # If we track the parity of A,B,C up to each prefix, a substring i..j (1-based)
    # is “good” if the parity-difference from i..j is (0,0,0) or (1,1,1), with
    # the odd case also requiring xA,xB,xC≥1 in that substring.  In prefix-parity
    # notation (0-based for code), that means:
    #
    #    P(j) xor P(i-1) in { (0,0,0), (1,1,1) },
    #    and if it is (1,1,1), then the substring includes at least one A,B,C.
    #
    # But the main difficulty is correlating how the '?'s are replaced so that
    # we can count how many total “good” substrings occur, and then how many
    # assignments produce at least K such good substrings.
    #
    # -----------------------------
    # A Viable DP over Prefixes
    # -----------------------------
    # A classic approach (for counting certain parity-based substring properties)
    # is to keep track of:
    #   dp[i][ s ][ g ] = number of ways to fill S[:i] (i characters) so that
    #       the parity-state at prefix i is s  (s in {0..7}, representing (a,b,c) mod 2),
    #       and we have exactly g “good” substrings so far.
    #
    # Then we move on to position i (0-based index in code).  If S[i] is fixed
    # as, say, 'B', we flip the 'B' bit in the old parity to get the new parity.
    # If S[i] is '?', we can flip any of the A/B/C bits.  We then must add
    # “inc” = the number of new good substrings that end exactly at i.  Each new
    # good substring that ends at i corresponds to a start-index j < i whose
    # parity state P(j) is either the same or the complement of the new parity
    # P(i).  In other words, if the new parity at i is x, then substring j+1..i+1
    # is good if P(j) in { x, x^7 }.  So the increment inc = the count of j in
    # [0..i-1] with parity in { x, x^7 } (plus possibly j=-1 if that leads to a
    # good substring from the very start).
    #
    # The catch: dp[i][s] does not tell us how many of those ways put P(j)= something
    # for each j.  It aggregates over many assignments.  We cannot simply multiply
    # sums, because “the same assignment that yields P(i)= s” might not always
    # produce P(j)= s' for j < i in consistent fashion.  Summing up counts of dp[j][s']
    # is double-counting across incompatible partial assignments.  The states
    # do not keep track of exactly which parities appeared at each intermediate
    # j.
    #
    # Thus a direct “dp[i][s][g] with a naive ‘inc’ = sum_{j} dp[j][s’ or comp(s’)]”
    # overcounts.  We need the number of j’s in the SAME path’s prefix that meet
    # the parity condition.
    #
    # -----------------------------
    # Storing the Parity-Frequency Distribution
    # -----------------------------
    # One way to resolve this is (in principle) to keep track of how many times
    # each parity state has occurred along the path, so when we add a new position
    # i with parity x, we can quickly see how many old positions j had parity x or x^7.
    # Then we can add that to the count of good substrings.  But that means we must
    # store an 8-dimensional distribution of how many positions mapped to each
    # of the 8 parity states.  That distribution’s size grows combinatorially with i,
    # making a naive DP too large for N=50.
    #
    # -----------------------------
    # Key Simplification: Parity “Pairs”
    # -----------------------------
    # Notice that for “goodness,” we only care if two parities are the same or complements
    # (x^7).  The 8 parity states can be grouped into 4 “pairs”:
    #   p0: {0,7}
    #   p1: {1,6}
    #   p2: {2,5}
    #   p3: {3,4}
    # Because 0^7 = 7, 1^6 = 7, 2^5 = 7, etc.  Moreover, a substring is good if
    # its endpoints are in the same or complementary state → that means “endpoints
    # are in the same pair.”  Also, flipping one bit moves you between specific pairs
    # in a predictable pattern:
    #   - Flip A-bit: p0 <-> p1, p2 <-> p3
    #   - Flip B-bit: p0 <-> p2, p1 <-> p3
    #   - Flip C-bit: p0 <-> p3, p1 <-> p2
    #
    # If along a single assignment path we keep track of how many prefix-indices
    # landed in each pair, when we set the new position i’s parity pair to pX,
    # the number of newly formed good substrings that end at i is exactly the
    # count of previous prefixes that are in that same pair pX.  (Because for
    # them to be “same or complementary in the 8-state sense” is the same as
    # being in the same pair, given our grouping.)
    #
    # So the increment inc = number_of_indices_in_that_path_so_far_that_are_in_pair(pX).
    #
    # But to do a DP over distributions (how many indices in p0, in p1, in p2, in p3)
    # plus the “current pair” plus “how many good substrings so far” can be quite large.
    # The dimension is i up to 50, c0+c1+c2+c3 = i+1 up to 51, each combination can be
    # ~ O(N^3), times 4 for the current pair, times up to 1275 possible good-substring
    # counts, leading to tens or hundreds of millions of states, which is borderline
    # or too large in Python.
    #
    # -----------------------------
    # Practical Implementation / Optimization
    # -----------------------------
    # Despite the large theoretical state space, with careful implementation,
    # pruning, and efficient dictionary-based or array-based transitions, one can
    # sometimes make it within time for N=50.  A key point is that many distributions
    # are not reachable, and also K can be up to N(N+1)/2 but we only need to store
    # dp states up to K (because once we exceed K, we can lump it into a “≥K” bin).
    #
    # We will implement the DP over:
    #   i in [0..N]  (how many characters processed),
    #   the counts (c0, c1, c2, c3) of how many prefix positions belong to each pair,
    #   the “current pair” curP of the prefix i,
    #   the number of good substrings so far (capped at K… if it exceeds K, we just store it as K+1 sentinel),
    # and dp[i][c0,c1,c2,c3][curP][g] = number of ways (mod 998244353).
    #
    # Then at the end, we sum over all (c0,c1,c2,c3,curP,g >= K) dp[N][...].
    #
    # Implementation details:
    # 1) We label the 4 pairs p0,p1,p2,p3.  We know position 0’s parity is 0 => that belongs to p0.  So
    #    initially i=0, we have (c0=1,c1=0,c2=0,c3=0, curP=0, g=0) = 1 way.
    # 2) When we process the i-th character of S (1-based for clarity, or 0-based carefully),
    #    we choose how it flips the old parity.  Actually the new index is i (1-based),
    #    so old “curP” is the pair of P(i-1).  We pick which letter? => we find the new 3-bit parity,
    #    from which we know the new pair newP.  Then the number of newly formed good substrings inc = the
    #    old count of how many prefixes were in newP.  That old count is precisely c[newP].  We add inc
    #    to g.  Then we increment c[newP] by 1 because position i also belongs to newP in the path.
    # 3) Because S[i-1] might be 'A','B','C', or '?', we handle either one flip or three possible flips.
    # 4) We cap g at K+1 so we don’t blow up memory with large g.  Because any state with g>=K can be lumped
    #    together as “already >=K.”  We only need to separate counts up to K-1 in detail, then everything
    #    beyond that merges into “>=K.”  At the very end, we will sum those “>=K” as well.
    #
    # The total number of states can be on the high side, but with N=50 it can typically still be done
    # in a carefully optimized Python (especially since many c0,c1,c2,c3 combos are not reachable).
    #
    # We will implement it using dictionary-based dynamic programming from layer i to i+1.  We store
    # only the states that have nonzero ways.
    #
    # Let us define how to map from a “pair” index to the actual 3-bit states, and how flipping bits
    # transitions between pairs:
    #
    #   We'll index pairs as:
    #       0 -> p0 = {0,7}
    #       1 -> p1 = {1,6}
    #       2 -> p2 = {2,5}
    #       3 -> p3 = {3,4}
    #
    #   Then define transitions for flipping 'A','B','C'.
    #   We'll define a small table nextPair[curPair][letter] => newPair.
    #
    # -----------------------------
    # Final Steps
    # -----------------------------
    # After building dp for i=N, we sum over all states where g >= K.  That sum mod 998244353 is our answer.
    #
    # Let’s implement it.
    #

    # First, precompute how flipping each letter moves us among the 8 parity states,
    # then reduce that to how it moves among the 4 pair-indices.

    # We'll number the 8 parity states as 0..7 in binary (cba):
    #   0 -> (0,0,0)
    #   1 -> (0,0,1)
    #   2 -> (0,1,0)
    #   3 -> (0,1,1)
    #   4 -> (1,0,0)
    #   5 -> (1,0,1)
    #   6 -> (1,1,0)
    #   7 -> (1,1,1)
    #
    # But we only track pairs:
    #   p0 -> {0,7}
    #   p1 -> {1,6}
    #   p2 -> {2,5}
    #   p3 -> {3,4}
    #
    # Let pairOfState = [ which pair each of the 8 states belongs to ]
    # Then define how flipping 'A' (xor 1), 'B' (xor 2), 'C' (xor 4) moves pairs.

    # Map each of the 8 states to its pair:
    pair_of = [0]*8
    # p0: {0,7}; p1: {1,6}; p2: {2,5}; p3: {3,4}
    pair_of[0] = 0
    pair_of[7] = 0
    pair_of[1] = 1
    pair_of[6] = 1
    pair_of[2] = 2
    pair_of[5] = 2
    pair_of[3] = 3
    pair_of[4] = 3

    # We also need, for each pair p in [0..3], and each letter L in {'A','B','C'},
    # the resulting new pair.  We'll pick a representative state s from pair p,
    # xor the bit for L => s' => find pair_of[s'] => that is new pair.
    #
    # For flipping bits: 'A'-> xor1, 'B'-> xor2, 'C'-> xor4.
    # We can choose one representative from each pair, e.g.:
    #   p0 rep = 0
    #   p1 rep = 1
    #   p2 rep = 2
    #   p3 rep = 3
    #
    # Then define transitions.  Because flipping from the other member of the pair
    # (e.g. 7 for p0) yields the same resulting pair as flipping 0 (for p0),
    # except we need to check it is indeed consistent.  It turns out that flipping
    # 0->1, 7->6 => pair_of(1)=1, pair_of(6)=1 => same pair.  So it’s consistent.

    rep_state = [0,1,2,3]  # representative states for p0,p1,p2,p3
    def flip(s, letter):
        if letter == 'A': return s ^ 1
        if letter == 'B': return s ^ 2
        return s ^ 4  # 'C'

    next_pair = [[0]*3 for _ in range(4)]
    letters = ['A','B','C']
    for p in range(4):
        srep = rep_state[p]
        for i,let in enumerate(letters):
            s2 = flip(srep,let)
            np = pair_of[s2]
            next_pair[p][i] = np

    # We'll implement a DP of the form:
    #
    #   dp[i] = dictionary of ((c0,c1,c2,c3, curP, g) -> ways)
    #
    # where i is how many positions we have assigned so far (0..N).  But i=0 is
    # the “prefix of length 0.”  We define that parity is state0=0 => pair p0.
    # Also c0=1 (because position “-1” we treat as parity index?), or we can
    # define we have exactly 1 prefix index:  that is the index = 0.  In many
    # parity-approaches we do include P(0) as the empty prefix’s parity.  Then
    # the first real character is i=1.  Let's do it carefully:
    #
    # Let us say:
    #   i in [0..N], i means we have placed i actual characters from S[0..i-1].
    #   So the # of prefix positions is i+1 if we count position 0 as the "empty prefix".
    #   The parity of the empty prefix is 0 => pair p0.  So initially dp[0] has:
    #       c0=1, c1=0, c2=0, c3=0, curP=0, g=0 => ways=1
    #
    # Then when we go from i to i+1, we are assigning S[i] (the (i+1)-th character).
    # For each old state, we pick which letter L we actually place (1 or 3 ways),
    # find new pair newP = next_pair[curP][ letterIndex], inc = oldCounts[newP],
    # newG = min(oldG + inc, K+1 to cap).  Then we increment that new state’s ways
    # in dp[i+1], but now cNewP = oldCounts[newP] + 1, and we adjust the distribution.
    #
    # Implementation details:
    #   - We store dp[i] as a dictionary key->val.  The key is (c0,c1,c2,c3,curP,g).
    #   - We iterate i from 0..N-1 to build dp[i+1].
    #   - In the end, we sum dp[N][...] over all keys whose g >= K.
    #
    # Because c0+c1+c2+c3 = i+1 is forced, we often do not need to store curP separately
    # if we want to re-construct it, but we do need it to identify transitions.  So we keep it.
    #
    # We must be mindful of efficiency in Python.  We’ll keep the next dp as a new dictionary,
    # and only proceed from states that have ways != 0.  We also cap g at K+1.  We store keys
    # in a well-structured tuple form.  The total number of reachable states is not too large
    # in practice for N=50 if done carefully.
    #

    # Precompute which letters are valid for each S[i].
    valid_letters = []
    for ch in S:
        if ch == '?':
            valid_letters.append([0,1,2])  # indices for 'A','B','C'
        elif ch == 'A':
            valid_letters.append([0])
        elif ch == 'B':
            valid_letters.append([1])
        else:  # 'C'
            valid_letters.append([2])

    # dp[i]: dictionary: (c0,c1,c2,c3, curP, g) -> ways
    from collections import defaultdict

    dp_current = defaultdict(int)
    # initial: i=0 => prefix of length 0 => we have 1 prefix index (the empty prefix),
    # which is in pair p0 => (c0=1,c1=0,c2=0,c3=0, curP=0, g=0)
    dp_current[(1,0,0,0, 0, 0)] = 1

    for i in range(N):
        dp_next = defaultdict(int)
        let_choices = valid_letters[i]

        for (c0,c1,c2,c3, curP, g), ways in dp_current.items():
            if ways == 0: 
                continue

            # We'll try each valid letter for S[i].
            for letter_idx in let_choices:
                newP = next_pair[curP][letter_idx]
                # inc = how many prior prefix positions were in pair newP
                # that is c0 if newP=0, c1 if newP=1, ...
                if   newP == 0: inc = c0
                elif newP == 1: inc = c1
                elif newP == 2: inc = c2
                else:           inc = c3

                newG = g + inc
                if newG > K: 
                    newG = K+1  # cap at K+1 to represent ">=K+1"

                # now increment the count for new distribution
                # we remove 1 from c[curP]?  Actually, no.  The old distribution c0,c1,c2,c3 
                # counts how many prefix positions so far are in each pair. 
                # We are adding a new prefix position i+1 in pair newP => so we do cNewP = cNewP+1.
                # The others remain the same.
                if   newP == 0: new_c0 = c0+1; new_c1=c1;   new_c2=c2;   new_c3=c3
                elif newP == 1: new_c0 = c0;   new_c1=c1+1; new_c2=c2;   new_c3=c3
                elif newP == 2: new_c0 = c0;   new_c1=c1;   new_c2=c2+1; new_c3=c3
                else:           new_c0 = c0;   new_c1=c1;   new_c2=c2;   new_c3=c3+1

                dp_next[(new_c0,new_c1,new_c2,new_c3, newP, newG)] = (dp_next[(new_c0,new_c1,new_c2,new_c3, newP, newG)] + ways) % MOD

        dp_current = dp_next

    # Now dp_current = dp[N], i.e. after assigning all N characters.  We want to sum
    # all states where g >= K.
    ans = 0
    for (c0,c1,c2,c3, curP, g), ways in dp_current.items():
        if g >= K:
            ans = (ans + ways) % MOD

    print(ans % MOD)