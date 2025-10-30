def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    t = int(input_data[0])
    # We'll parse the input three lines at a time for each test case
    # Format per test case: S, X, Y

    # ----------------------------------------------------------------
    # IDEA / EXPLANATION of the APPROACH:
    #
    # We want to decide if there exists a string T (possibly empty)
    # such that f(S,T,X) = f(S,T,Y), where
    #   f(S,T,X) = concatenation of [S if X[i]=='0' else T if X[i]=='1']
    #
    # Observing that f(S,T,X) is made up of blocks of S and T in the
    # exact order given by X, and likewise for Y:
    #
    #   f(S,T,X) = (S or T) + (S or T) + ... (|X| times),
    #   f(S,T,Y) = (S or T) + (S or T) + ... (|Y| times).
    #
    # For these two resulting strings to be identical, two conditions
    # must be met:
    #
    # 1) They must have the same length.  That is:
    #    (#0_X * |S| + #1_X * |T|) == (#0_Y * |S| + #1_Y * |T|)
    #    where #0_X is the count of '0' in X, #1_X is the count of '1' in X,
    #    #0_Y is the count of '0' in Y, #1_Y is the count of '1' in Y.
    #
    #    Let:
    #       A = ( #0_Y - #0_X ) * len(S)
    #       B = ( #1_X - #1_Y )
    #    Then we need B * len(T) = A.
    #
    #    - If B=0, then #1_X = #1_Y. For the lengths to match, we also need
    #      #0_X = #0_Y. Hence X and Y must have exactly the same number of '0's
    #      and '1's. If that is not true, no T can fix the length mismatch.
    #
    #    - If B != 0, then we need len(T) = A / B as an integer ≥ 0. That is,
    #      A must be divisible by B, and the quotient must be ≥ 0. If that
    #      fails, we can say "No" immediately.
    #
    # 2) Even if the lengths match, the actual strings must match
    #    character by character.  We must ensure that when we expand
    #    f(S,T,X) and f(S,T,Y), they can align perfectly as the same string.
    #
    #    Key observation: S is fully known, T is unknown. When we see a
    #    run of zeros from X versus a run of zeros from Y, that part
    #    expands to repeated copies of S in both final strings, so that
    #    segment trivially matches. Similarly, a run of ones in X or Y
    #    expands to repeated T. If T is the same string on both sides,
    #    then that part also trivially matches. Thus the only potential
    #    problem is when one side says "0" (meaning S) and the other side
    #    says "1" (meaning T). Then we are forced to match S with some
    #    portion of T (or vice versa). This imposes a strong constraint
    #    on what T can be.
    #
    #    In particular, any time a piece of S needs to match a piece of T,
    #    it forces T (or that piece of T) to be exactly S (or to contain S
    #    as a prefix, etc.). Repeated occurrences of such mismatches
    #    must remain consistent. If we find two contradictory forced
    #    assignments on T, we must say "No".
    #
    # However, there is a powerful simplifying fact:
    # ------------------------------------------------
    #    If X and Y ever differ in which bits are 0/1 at the same
    #    "position" in the final string, that mismatch forces S = T or
    #    T = S or that S is a prefix of repeated T or T is a prefix of
    #    repeated S in a way that must remain consistent throughout.
    #
    #    But carefully verifying all partial overlaps can be very large
    #    if done naively.  A known simpler method is:
    #
    #         "Scan X and Y in parallel (a 'two-pointer' over the bits),
    #          expanding them into the final string in a synchronized
    #          manner. Whenever both are about to append the same type
    #          of block (both 0 or both 1), that part matches trivially.
    #          Whenever they differ, we incrementally check that the entire
    #          block of S matches the entire block(s) of T or vice versa.
    #          If ever inconsistent, answer No."
    #
    # But there's an even more direct short-circuit:
    #
    #    - A mismatch of 0 vs 1 means we must match an S-block with a T-block.
    #      Because S is known, for that block to match, T must be exactly
    #      that same substring (or repeated concatenations, if one side
    #      covers multiple blocks). If such enforced T is contradictory to
    #      a previously enforced T, we fail immediately.
    #
    #    - If we manage to get to the end consistently, we succeed.
    #
    # Implementation sketch for large constraints:
    #
    # We'll do this in two phases:
    #   Phase A: Check if length matching is *possible*. (The B*len(T)=A check.)
    #   Phase B: If length matching is not possible -> "No".
    #            Else, we do an efficient "run-length" style two-pointer
    #            to ensure expansions can match.  We'll handle forced
    #            S<->T matches by requiring T to equal S if that ever arises.
    #            If we ever get forced T=non-empty-S and T=some-other-string
    #            that doesn't match, we say "No".
    #
    # There is a corner case if T is forced to be empty.  That is valid if
    # len(T) = 0 satisfies B*0=A => A=0 => implies #0_X = #0_Y and #1_X= #1_Y
    # so that it doesn't break expansions.  In that scenario, all 1-bits
    # expand to the empty string, so we just check if the S-block expansions
    # match.  That is consistent if and only if X's 0-bits pattern matches
    # Y's 0-bits pattern in length.
    #
    # After ensuring the lengths possibly match, we do:
    #
    #    1) If #1_X = 0 and #1_Y = 0, that means no T is used at all.
    #       Then we just need to check if X and Y produce the same repetition
    #       of S => i.e. #0_X == #0_Y so that they produce the same final
    #       string = S repeated. If that is the case -> "Yes", else -> "No".
    #
    #    2) If T is forced to be empty (from the length equation) then check
    #       if that final string equals or not.
    #
    #    3) Otherwise, we do the run-length decomposition approach on X and Y
    #       and try to match expansions.  We'll only need to check the
    #       cross 0<->1 blocks.  If we are forced S = T (non-empty) in some
    #       mismatch, that means T must be exactly S.  If we get forced
    #       T = "" from the equation, that is also a possibility.  If we
    #       get forced T = S in one mismatch but T = something else later,
    #       we say "No".
    #
    # A simpler shortcut often emerges in problems like this:
    #
    #   "If there's ever a position in X's expansion that is S and the
    #    corresponding position in Y's expansion is T, then T must match S
    #    there.  Because S is known, T is forced. If we see *another*
    #    mismatch at a *different* spot also forcing T to match a *different*
    #    substring of S, we'd get a contradiction unless that substring is
    #    the same as S.  Summarizing: either T is empty, or T = S, or no
    #    solution.  Because partial overlaps quickly force T to be exactly
    #    S if T is non-empty."
    #
    # Indeed, one can show that if T must match S even once in these expansions
    # (and T is non-empty), we are forced T = S globally in order to remain
    # consistent everywhere.  Therefore, the only two possibilities are:
    #
    #   (a) T is empty.
    #   (b) T = S.
    #
    # This drastically simplifies the logic:
    #
    #   1) Check if T=empty works: 
    #      Then f(S,"",X) is just S for each '0' in X and "" for each '1' in X.
    #      We get the final string = S repeated (#0_X times).  Similarly for Y
    #      => S repeated (#0_Y times).  They match if and only if #1_X=0 and
    #      #1_Y=0 (so that the '1' bits don't appear) and #0_X = #0_Y.  If that
    #      is satisfied -> "Yes".  Otherwise, T=empty doesn't solve it.
    #
    #   2) Check if T=S works:
    #      Then f(S,S,X) = for each bit in X, we append either S (if 0) or S (if 1),
    #      i.e. always S.  So f(S,S,X) is just S repeated len(X) times.
    #      Similarly, f(S,S,Y) is just S repeated len(Y) times.
    #      They match if and only if len(X) = len(Y).  If that is satisfied -> "Yes".
    #
    #   3) If neither T=empty nor T=S works, answer "No".
    #
    # That is indeed enough.  The examples confirm it:
    #   - In sample1, test1: T was "ara", but observe T= S? Actually S="araara".
    #     Wait, it looks contradictory. But let's see the expansions:
    #       X="01" => f(S,T,X) = S + T
    #       Y="111" => f(S,T,Y) = T + T + T
    #     They matched for T="ara", S="araara". Notice T != S. 
    #     But "ara" turned out to be exactly the "root" of S? Actually "araara"
    #     is "ara" repeated 2 times minus overlap? It’s tricky. Let's check carefully:
    #       S="araara", T="ara"
    #       S repeated once = "araara"
    #       T repeated once = "ara"
    #     S is not the same as T. They are not equal strings. "ara" vs "araara".
    #
    # So the claim "T must be empty or T = S" is not always correct by naive reading.
    #
    # The trick, however, is that "araara" is itself "ara" repeated 2 times with overlap?
    # Actually "araara" = "ara" + "ara", straightforward.

    # But let's see what's happening with expansions:
    #   f(S,T,X="01") = S + T = "araara" + "ara" = "araaraara"
    #   f(S,T,Y="111")= T + T + T = "ara" + "ara" + "ara" = "araaraara"
    # They matched. So T != S, but S = T repeated 2 times.  Indeed S="araara" = "ara" * 2.
    #
    # So more precisely, every mismatch 0->1 or 1->0 will force that *one* block
    # (S or T) to be some power-of the other block.  i.e. S is k copies of T
    # or T is k copies of S.  And if multiple mismatches occur, we must preserve
    # consistency.  In the example, S is exactly T repeated 2 times. So whenever
    # we need S to match T or T to match S, that can remain consistent if S is
    # a repetition of T or T is a repetition of S (the same base pattern).
    #
    # So the underlying condition is: S and T must share the same "primitive root"
    # (the smallest string r such that S=r^p, T=r^q for some p,q>0). Then all
    # S<->T mismatches can match up with the appropriate repetition count.
    #
    # Implementation steps:
    #  (A) Check length condition. If impossible, "No".
    #  (B) Let rS = the primitive root of S.  For existence of T,
    #      we also guess T shares that same root rS. Then T = rS^k for some k≥0.
    #      Or T could be empty (special case). 
    #      So the entire expansions can match only if every mismatch of bits
    #      can be explained by S being some integer multiple of rS and T being
    #      some (possibly different) integer multiple of rS.  Since we already
    #      know S = rS^(len(S)//len(rS)), T must be rS^k.
    #
    #      We only need to figure out k = len(T)//len(rS). From the length equation,
    #      if B != 0, len(T)=A/B => we must check that len(T) is multiple of len(rS).
    #      Then T = rS^( len(T)/len(rS) ).
    #
    # (C) Now we do a final "two-pointer" check of expansions where
    #     S-block = rS^( some p ) and T-block = rS^(some q ) to ensure that
    #     the expansions match if we line them up. Actually if we have
    #     the correct length for T and S and they share the same root rS,
    #     then the expansions will match as long as the bit patterns
    #     line up with the same root string. The only subtlety is partial
    #     blocks. But since S is a whole number of rS and T is a whole number
    #     of rS, concatenating them in any order still forms a big repetition
    #     of rS. Similarly for the other side. So they will match if we have
    #     the same total length and the same root. 
    #
    # In short, the final check is:
    #   1) B * len(T) = A => check divisibility => possible T length >= 0
    #   2) If len(T)>0, let gcd_root = the gcd-string of S. If T is to share
    #      that root, len(T) must be a multiple of len(gcd_root). Then define
    #      T = gcd_root^( len(T)//len(gcd_root) ).
    #      Also S must be gcd_root^( len(S)//len(gcd_root) ).
    #      Then the expansions are all in the language of gcd_root repeated,
    #      so indeed expansions match as soon as the lengths line up. => "Yes"
    #
    #      If len(T)=0, we just check if the expansions match (that means #1_X=0
    #      and #1_Y=0 for any non-empty S, and #0_X=#0_Y, or S must be repeated
    #      the same number of times).
    #
    # So the final simpler formula is:
    # --------------------------------------------------------------------
    # function primitive_root(str) -> the smallest "r" such that str = r^(some power)
    #
    # Steps for each test:
    #   (1) read S, X, Y
    #   (2) count0_X, count1_X, count0_Y, count1_Y
    #   (3) check length feasibility:
    #       B = (count1_X - count1_Y)
    #       A = (count0_Y - count0_X)*len(S)
    #       if B=0:
    #         if A!=0 -> "No"
    #         else we have #0_X=#0_Y and #1_X=#1_Y
    #              then we'd still need to check expansions. But in fact,
    #              the expansions are both (#0_X times S) + (#1_X times T).
    #              They have same length. We only need to check if there's
    #              a T that can unify them. If #1_X=0 => then T never appears:
    #                check #0_X=#0_Y => "Yes" else "No".
    #              Otherwise #1_X>0 => T must appear in expansions. If there's
    #              also a mismatch in bits order, T might get forced to match S
    #              or S might get forced to match T, requiring same root. So we
    #              attempt the root-check approach with T of any length. Then
    #              an infinite number of T's might match? Possibly we only need
    #              to see if S has some root rS. Then any T = rS^k also shares
    #              the same root => expansions become rS repeated. So "Yes".
    #       else:
    #         if A % B != 0 -> "No"
    #         else lenT = A//B, must be >= 0:
    #              if lenT<0 -> "No" else feasible length for T
    #
    #   (4) If feasible, do the "common root" check:
    #       let rS = primitive_root(S). Then:
    #         - if lenT>0, check lenT % len(rS)==0. If not, "No".
    #         - else if lenT=0, that is a special case we allow if it doesn't
    #           conflict with the expansions' bit usage. (In particular, if #1_X>0
    #           or #1_Y>0, we do produce T-block which is empty string. That
    #           can still match as empty concatenations. So it might be okay.)
    #       - also check that S is (rS repeated). If not, "No". (We glean rS from S
    #         so that is guaranteed.)
    #       - If all good => "Yes", else => "No".
    #
    # Let's code it.
    #
    # Complexity: computing primitive root of S is O(|S|).  Summed over all tests
    # up to 5e5, so we must do that carefully. We'll do a standard "prefix-function"
    # approach to find the minimal period of S.  Then the rest is just quick checks.
    #
    # Implementation details next:
    # ----------------------------------------------------------------

    # A helper to compute the length of the primitive root of a string
    # using the prefix-function (KMP) trick:
    def primitive_root_length(st):
        # prefix-function (pi) approach
        n = len(st)
        pi = [0]*n
        # build prefix function
        # standard O(n) prefix-function computation:
        i, j = 1, 0
        while i < n:
            while j > 0 and st[i] != st[j]:
                j = pi[j-1]
            if st[i] == st[j]:
                j += 1
                pi[i] = j
            i += 1
        # the length of the smallest period:
        # if n % (n - pi[n-1]) == 0, that period is (n - pi[n-1]), else n
        period = n - pi[n-1]
        if n % period == 0:
            return period
        else:
            return n

    # We'll read index from input_data:
    idx = 1

    # Precompute prefix function / root length of S for each test might be large,
    # but we have to do it once per test anyway. We'll do it carefully.

    # We store results in a list to output at the end (faster I/O).
    answers = []
    out = []

    for _ in range(t):
        S = input_data[idx]; idx+=1
        X = input_data[idx]; idx+=1
        Y = input_data[idx]; idx+=1

        # Count 0 and 1 bits in X, Y
        count0_X = X.count('0')
        count1_X = len(X) - count0_X
        count0_Y = Y.count('0')
        count1_Y = len(Y) - count0_Y
        lenS = len(S)

        # Let B = (count1_X - count1_Y)
        #     A = (count0_Y - count0_X)*len(S)
        B = count1_X - count1_Y
        A = (count0_Y - count0_X)*lenS

        # Quick check if B=0 => either A=0 or No
        if B == 0:
            if A != 0:
                # No, because we can't match lengths
                out.append("No")
                continue
            # So #1_X = #1_Y and #0_X = #0_Y => the expansions have the same total length
            # We just need to see if there's some T that can make
            # the expansions identical in structure. Let's see:
            #
            # If #1_X=0 => expansions are S repeated #0_X times => they'd match
            # if #0_X is the same => indeed #0_X is the same. => "Yes".
            # If #1_X>0 => we do have T-blocks. Potentially many T's could work
            # as long as S and T share a common root. But the question is always
            # "does some T exist at all?" The answer is "Yes" because we can pick
            # T = S (or T = empty if #1_X>0 is not allowed, but T=S definitely
            # will produce the same expansions if the bit patterns are the same
            # count of 0 and 1). Actually if X=Y exactly as bit strings, f(S,T,X)
            # matches f(S,T,Y) for any T. If X != Y but have same # of 0 and # of 1,
            # the order might differ, forcing partial overlaps. Yet we can pick
            # T such that T is the gcd-root with S so that partial S <-> T collisions
            # are consistent. There always is such a T (worst case T = S repeated
            # enough times). Because the final string is a puzzle of S and T blocks
            # that can be rearranged to the same big repetition of the gcd-root.
            #
            # Conclusion: If #0_X=#0_Y and #1_X=#1_Y, answer "Yes".
            out.append("Yes")
            continue

        # Now B != 0 => we want len(T) = A / B integer >=0
        if A % B != 0:
            out.append("No")
            continue
        lenT = A // B
        if lenT < 0:
            out.append("No")
            continue

        # Now we have a candidate length for T. If lenT=0, T is the empty string.
        # Then f(S,T,X) = S repeated #0_X times, f(S,T,Y) = S repeated #0_Y times.
        # For them to match, we need #0_X = #0_Y and also all '1's produce empty
        # strings, which does not break anything automatically. So let's check:
        if lenT == 0:
            # expansions match if and only if S repeated #0_X times = S repeated #0_Y times
            if count0_X == count0_Y:
                out.append("Yes")
            else:
                out.append("No")
            continue

        # If lenT>0, we want to ensure that S and T share the same primitive root
        # so that any mismatch of blocks can align. So let's compute the root of S.
        prSlen = primitive_root_length(S)
        # The "root" string rS
        rS = S[:prSlen]
        # Check that S is indeed a repetition of rS
        if lenS % prSlen != 0:
            out.append("No")
            continue
        # Verify S == rS^( lenS//prSlen )
        if rS * (lenS//prSlen) != S:
            out.append("No")
            continue

        # Next check lenT is multiple of len(rS)
        if lenT % prSlen != 0:
            # Then T can't be an integer repetition of rS
            out.append("No")
            continue

        # If that is fine, T = rS^(lenT//prSlen), so expansions in either X or Y
        # will produce a big repetition of rS, all of which line up in the
        # same way as long as the total lengths match. We already matched lengths,
        # so it should be "Yes".
        out.append("Yes")

    print("
".join(out))