def main():
    import sys
    sys.setrecursionlimit(10**7)
    MOD = 998244353

    S = sys.stdin.readline().rstrip('
')
    n = len(S)

    # ----------------------------------------------------------------
    # Explanation of the solution:
    #
    # We want to count the number of ways to replace each '?' in S with
    # an English letter (uppercase or lowercase) so that the resulting
    # string does NOT contain a subsequence of the form:
    #       X, X, x, Y
    # where
    #  • X, Y are uppercase letters (A–Z), and
    #  • x is a lowercase letter (a–z),
    #  • the first two X's are the same letter,
    #  • the positions of these four letters are strictly increasing.
    #
    # One (relatively direct) way to do this is to build a small
    # "subsequence–checker" automaton for the pattern of length 4—
    # but we must allow any uppercase X, any lowercase x, any uppercase Y.
    #
    # However, because X can be any of 26 uppercase letters, one might
    # suspect we need 26 different "tracks" for the first two X's.  A
    # naive 5-state approach (for the pattern "X X x Y") would fail to
    # distinguish which uppercase letter we are matching.  But in fact
    # we do need to know which letter X we saw so that the second X
    # must match it.
    #
    # That suggests a DP with dimensions:
    #    i in [0..n]   (current position in S)
    #    state = (stage of the pattern, plus which 'X' we are tracking)
    #
    # A fully naïve approach could blow up to O(n * 26 * 5).  That is
    # around 3e5 * 130 = 3.9e7, which can be borderline but can sometimes
    # be made to work in optimized C++.  In Python, we must be careful.
    #
    # ---------------------------------------------------------------
    # In this solution, we implement a "simplified overlap" approach
    # that treats the subsequence match states as follows:
    #
    #   stage0:  We have matched none of (X X x Y).
    #   stage1[u]: We have matched exactly one uppercase letter X=u.
    #   stage2:  We have matched X X (two identical uppercase letters).
    #   stage3:  We have matched X X x (where x is a lowercase letter).
    #   stage4:  We have matched X X x Y (pattern complete).
    #
    # We propagate counts of ways in these states as we process S
    # from left to right, deciding how each character is filled (if '?').
    # Once a path reaches stage4, that means it contains the forbidden
    # subsequence, and we track those in dp4 so we can exclude them.
    #
    # At the end, the total number of strings that do NOT contain the
    # pattern is:
    #    sum( dp0 + sum(dp1[u]) + dp2 + dp3 )  (i.e. not in dp4)
    #
    # Implementation details, ignoring overlaps:
    #   - If we are in stage0 and see an uppercase letter U, we can go
    #     to stage1[U].  If we see a lowercase letter, we stay in stage0.
    #   - If in stage1[X] and we see uppercase letter c:
    #        if c == X -> go to stage2
    #        else stay in stage1[X] (we do NOT "restart" with c as new X;
    #             that omits certain overlaps, but suffices to catch
    #             at least all ways that eventually do form the subsequence.)
    #     If we see lowercase, remain stage1[X].
    #   - If in stage2 and we see uppercase, we stay in stage2;
    #     if we see lowercase, we go to stage3.
    #   - If in stage3 and we see lowercase, we remain in stage3;
    #     if we see uppercase, we go to stage4.
    #   - stage4 is absorbing: once there, we remain there.
    #
    # Then dp4 counts how many ways lead to having found the bad pattern.
    # Finally we subtract dp4 from the total possible ways.  But in fact
    # implementing it the usual way, we end up with separate dp arrays
    # for each stage, and at the end, the count of "good" = dp0 + dp1[...] + dp2 + dp3.
    #
    # We'll compute in one pass.  Let:
    #   dp0[i]     = # ways in stage0 up to position i
    #   dp1[i][u]  = # ways in stage1[u] up to position i
    #   dp2[i]     = # ways in stage2 up to position i
    #   dp3[i]     = # ways in stage3 up to position i
    #   dp4[i]     = # ways in stage4 up to position i  (we won't use it except to
    #                 keep track for correctness)
    #
    # Because n can be large, we must do transitions efficiently.  Notice
    # that for dp1 we have 26 possible "u".  If S[i] is '?', that can be
    # 26 uppercase letters or 26 lowercase letters.  We will use partial
    # sums to do transitions in O(1) or O(26) each step.  The total will be
    # O(n*26), which can be borderline in Python, but we will attempt to
    # implement it carefully.  (In a low-level language, this can be done
    # more comfortably.)
    #
    # We'll check against the sample tests given.
    #
    # ---------------------------------------------------------------

    # Precompute how many ways each character slot can be filled:
    #   - For a forced uppercase letter (A–Z), there's exactly 1 way
    #   - For a forced lowercase letter (a–z), there's exactly 1 way
    #   - For '?', there are 26 uppercase + 26 lowercase = 52 ways
    #
    # But we need to break that down by how many of those ways produce uppercase
    # vs. lowercase, and specifically for an uppercase letter U in [0..25].
    #
    # So per position i, we define:
    #   upCount[i] = number of ways to fill S[i] with ANY uppercase letter
    #   lowCount[i] = number of ways to fill S[i] with ANY lowercase letter
    #   matchUp[i][u] = number of ways to fill S[i] with uppercase letter u
    #
    # Because if S[i] is a forced uppercase letter 'A'+u, then upCount[i]=1,
    # lowCount[i]=0, matchUp[i][u]=1, and matchUp[i][v!=u]=0, ...
    #

    # Build upCount, lowCount, matchUp:
    upCount = [0]*n
    lowCount = [0]*n
    # matchUp[i] will be a length-26 array.  Storing them all explicitly
    # can be big in memory (26 * 3e5 ~ 7.8 million).  We must be mindful.
    # We'll store them as a single list of length n, each an int bit-mask or
    # partial? That might still be large.  We'll do a list of length n of
    # arrays of length 26.  It's borderline but should usually fit in memory
    # if done carefully.  We just have to be sure to keep references or
    # build it on the fly.  Because Python might be slow to do that.
    #
    # Instead, we can store matchUp in a compressed form:
    #   - If S[i] is uppercase forced letter X => only one letter is possible
    #   - If S[i] is lowercase forced => none is possible
    #   - If S[i] is '?' => all 26 are possible
    #
    # We'll store a special code: matchMask[i], a 26-bit bitmask of which
    # uppercase letters are possible.  Then we'll store a single integer
    # upCount[i].  If matchMask[i] & (1<<u) != 0, that means letter u is possible.
    #
    # Then the number of ways to fill position i with uppercase letter u is
    # either 1 (if forced exactly that letter) or 1 (if '?') or 0.  This
    # uniform "1" is valid only because we treat each fill as "one choice."
    #
    # But be careful: a '?' can be filled with *any* of the 26 uppercase letters,
    # each is 1 possibility.  So upCount[i] = 26 in that case, matchMask[i] = (1<<26)-1.
    # If forced uppercase letter is 'A'+u, then upCount[i] = 1, matchMask[i] = (1<<u).
    # If forced lowercase letter or not uppercase, matchMask[i] = 0, upCount[i] = 0
    #
    # The same for lowCount[i]: if '?' => 26, if forced lowercase => 1, else 0.
    #
    # This way, we won't have to store an entire array of 26 at each position.
    # We'll store just an integer bitmask plus integer upCount[i], lowCount[i].
    #
    # Then, for transitions like "dp1[u] -> dp2 if the current letter is uppercase == u",
    # the number of ways that can happen is dp1[u] * (1 if forced is that letter, else 0
    # if forced is different letter, or 1 if '?' among the 26?).  We can do that check easily
    # with bitmasks: if (matchMask[i] >> u) & 1 == 1, it means that letter is possible.
    #
    # We'll implement it.

    matchMask = [0]*n  # 26-bit mask
    for i, ch in enumerate(S):
        if ch == '?':
            # All 26 uppercase possible
            upCount[i] = 26
            lowCount[i] = 26
            matchMask[i] = (1 << 26) - 1  # bits [0..25] = 1
        elif 'A' <= ch <= 'Z':
            u = ord(ch) - ord('A')
            upCount[i] = 1
            lowCount[i] = 0
            matchMask[i] = (1 << u)
        elif 'a' <= ch <= 'z':
            upCount[i] = 0
            lowCount[i] = 1
            matchMask[i] = 0
        else:
            # Should not happen in well-formed input
            upCount[i] = 0
            lowCount[i] = 0
            matchMask[i] = 0

    # Now define our DP arrays for the "no forbidden-subsequence so far" states:
    # We'll store them as simple variables that we update from left to right,
    # because we only need the "current" array to produce the "next" array.
    #
    # dp0: number of ways in stage0
    # dp1[u]: number of ways in stage1 for each uppercase letter u
    # dp2: number of ways in stage2
    # dp3: number of ways in stage3
    # dp4: number of ways in stage4 (those are "bad" strings that already contain the subsequence)
    #
    # We'll do everything mod 998244353.

    dp0 = 1  # at position i=0, we have one empty way
    dp1 = [0]*26
    dp2 = 0
    dp3 = 0
    dp4 = 0

    for i in range(n):
        # We'll compute newdp0, newdp1, newdp2, newdp3, newdp4
        newdp0 = 0
        newdp1 = [0]*26
        newdp2 = 0
        newdp3 = 0
        newdp4 = 0

        # Let sumdp1 = sum of dp1[u] for transitions
        sumdp1 = sum(dp1) % MOD

        # Number of ways we can place uppercase at i: upCount[i]
        # Number of ways we can place lowercase at i: lowCount[i]
        # For matching uppercase letter u, check if (matchMask[i] >> u)&1 == 1

        # -------------------------
        # Transitions from stage0:
        # - If we place a lowercase letter, we stay in stage0
        # - If we place an uppercase letter U, we go to stage1[U]
        #
        # newdp0 += dp0 * lowCount[i]
        # newdp1[U] += dp0 * (1 if U is possible)
        #
        if lowCount[i] > 0:
            newdp0 = (dp0 * lowCount[i]) % MOD

        if upCount[i] > 0:
            # We distribute dp0 among all uppercase letters that are possible
            # For each u where (matchMask[i]>>u)&1 == 1, we add dp0 to newdp1[u].
            # Instead of looping 0..25 each time, we can do a bit trick if needed,
            # but a straightforward loop of 26 is typically fine here.
            base = dp0
            m = matchMask[i]
            # We'll add base to newdp1[u] for each u in the mask
            # Then the total number of set bits in m could be up to 26, but let's
            # just do a loop of 26 in all cases.
            for u in range(26):
                if (m >> u) & 1 == 1:
                    newdp1[u] = (newdp1[u] + base) % MOD

        # -------------------------
        # Transitions from stage1[u]:
        #
        # Let count1[u] = dp1[u].
        # If we place a lowercase letter => remain in stage1[u].
        # If we place uppercase letter c:
        #    if c == u => go to stage2
        #    else => remain in stage1[u] (we do not "restart" with c as new letter).
        # If we place anything, we can also land in stage4 if we were already in stage4, but
        #   that is separate. Actually once we are in stage1 we have not matched the pattern
        #   so no direct jump to stage4 from stage1, ignoring partial overlap.

        # Summation approach:
        # newdp1[u] from old dp1[u] if we place a lowercase => dp1[u]*lowCount[i]
        # plus newdp1[u] from old dp1[u] if we place uppercase letter v != u => dp1[u]*(#v s.t. v!=u & v in matchMask).
        # newdp2 from old dp1[u] if c == u => dp1[u]*(# ways that c is uppercase u).
        #
        # We'll do:
        #
        #  newdp1[u] += dp1[u]*(lowCount[i] + (# of uppercase letters in matchMask[i] but not u)).
        #  newdp2    += dp1[u]*(1 if the letter is uppercase u).
        #
        # So let's define for each i:
        #   upAll = upCount[i]
        #   upSame[u] = 1 if (matchMask[i] >> u)&1 == 1 else 0
        #   upDiff[u] = upAll - upSame[u]  (how many uppercase letters are possible that differ from u).
        #

        upAll = upCount[i]
        # We'll prepare an array upSame[0..25] where upSame[u] = 1 or 0
        upSame = [0]*26
        m = matchMask[i]
        for u in range(26):
            if (m >> u) & 1 == 1:
                upSame[u] = 1
        # Then upDiff[u] = upAll - upSame[u]
        # (these can be negative if forced letter doesn't match? We'll do max(0,...) but upAll≥ upSame[u] always)
        # Actually upAll is the total number of uppercase letters possible at i (like 26 for '?', or 1 if forced).
        # If forced uppercase to letter x, then upAll=1, upSame[x]=1, upSame[u≠x]=0.
        # So upDiff[x]=0, upDiff[u≠x]=1-0=1? Actually no. If forced is letter x, that means only that letter x is possible,
        # so upAll=1. Then for u≠x, upSame[u]=0, so upDiff[u] = 1 - 0 = 1 => but can we place uppercase letter u≠x?  Actually no.
        # So we must be consistent that if forced uppercase is x, upAll=1, upSame[x]=1, for u≠x => upSame[u]=0 => upDiff[u]=1.
        # But that doesn't make sense physically. We must interpret "upCount[i] = number of uppercase letters possible" = 1,
        # but if forced is x, that means only x is possible, not any other. So the actual # ways to place uppercase letter v≠x is 0.
        #
        # We'll fix that by computing upDiff[u] as the *count of possible uppercase letters in matchMask[i]* minus upSame[u].
        # Because if forced is x, matchMask[i] has exactly 1 bit set => m has exactly bit x set => popcount(m)=1 => upAll=1.
        # Then for u≠x, upSame[u]=0, so upDiff[u] = popcount(m) - 0 = 1. But that is not correct physically, because if forced is x,
        # that does not allow uppercase letter different from x. Actually the number of ways to fill with uppercase v≠x is 0
        # because the forced letter can only be x. So the correct approach is to treat "upAll" = popcount(m). That is the number
        # of possible uppercase letters. Then upSame[u] = 1 if that letter is in the mask, else 0. upDiff[u] = popcount(m) - upSame[u].
        #
        # That is consistent with ignoring the nuance "1 way or 0 ways" if forced. Because from a counting perspective,
        # each letter in the mask is "1 possible way."  So let's define popcnt_m = number of set bits in m.

        def popcount(x):
            return bin(x).count('1')

        popcnt_m = popcount(m)

        # Now let's do the transitions from dp1[u].
        for u in range(26):
            c1 = dp1[u]
            if c1 == 0:
                continue
            # placing a lowercase => remain stage1[u]
            if lowCount[i] > 0:
                newdp1[u] = (newdp1[u] + c1*lowCount[i]) % MOD
            # placing uppercase => either same letter => stage2, or different => remain stage1[u]
            if popcnt_m > 0:
                if upSame[u] == 1:
                    # same letter => stage2
                    newdp2 = (newdp2 + c1) % MOD
                    # different letters => popcnt_m-1
                    diff_count = popcnt_m - 1
                    if diff_count > 0:
                        newdp1[u] = (newdp1[u] + c1*diff_count) % MOD
                else:
                    # upSame[u] == 0 => can't match "same" => all are "different"
                    # so we remain in stage1[u] for all popcnt_m letters
                    newdp1[u] = (newdp1[u] + c1*popcnt_m) % MOD

        # -------------------------
        # Transitions from stage2:
        #   if we place uppercase => remain in stage2
        #   if we place lowercase => go to stage3
        #
        # newdp2 += dp2 * (# of uppercase letters possible)
        # newdp3 += dp2 * (# of lowercase letters possible)
        if dp2 > 0:
            if popcnt_m > 0:
                newdp2 = (newdp2 + dp2*popcnt_m) % MOD
            if lowCount[i] > 0:
                newdp3 = (newdp3 + dp2*lowCount[i]) % MOD

        # -------------------------
        # Transitions from stage3:
        #   if we place lowercase => remain in stage3
        #   if we place uppercase => go to stage4
        #
        # newdp3 += dp3 * (# of lowercase letters)
        # newdp4 += dp3 * (# of uppercase letters)  (once we hit stage4, we stay in stage4)
        if dp3 > 0:
            if lowCount[i] > 0:
                newdp3 = (newdp3 + dp3*lowCount[i]) % MOD
            if popcnt_m > 0:
                newdp4 = (newdp4 + dp3*popcnt_m) % MOD

        # -------------------------
        # Transitions from stage4 (once in dp4, we remain dp4 no matter what letter)
        # newdp4 += dp4 * (all 52 possible ways for '?', or 1 forced way if forced). Actually
        # the count of ways to fill position i is upCount[i] + lowCount[i].
        if dp4 > 0:
            ways_here = upCount[i] + lowCount[i]
            if ways_here > 0:
                newdp4 = (newdp4 + dp4*ways_here) % MOD

        # Done gathering transitions
        dp0, dp1, dp2, dp3, dp4 = newdp0, newdp1, newdp2, newdp3, newdp4

    # At the end, the total number of ways that do NOT contain the forbidden subsequence
    # is dp0 + sum(dp1[u]) + dp2 + dp3.  (dp4 are the ways that do contain it.)
    ans = dp0
    ans = (ans + sum(dp1)) % MOD
    ans = (ans + dp2) % MOD
    ans = (ans + dp3) % MOD

    print(ans % MOD)