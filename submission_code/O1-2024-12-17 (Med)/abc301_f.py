def main():
    import sys
    sys.setrecursionlimit(10**7)
    MOD = 998244353

    S = sys.stdin.readline().rstrip('
')
    n = len(S)

    # We will count (mod 998244353) the number of ways to replace all '?' such that
    # NO subsequence of the form "X X x Y" appears, where:
    #   - X is an uppercase letter (the same letter repeated for the 1st and 2nd positions of the subsequence),
    #   - x is a lowercase letter,
    #   - Y is an uppercase letter.
    #
    # Approach:
    # Let A0 = number of ways (so far) with no partial match,
    #     A1[x] = number of ways with exactly one uppercase letter matched (which is 'A'+x),
    #     A2[x] = number of ways with two identical uppercase letters matched (X X),
    #     A3[x] = number of ways with X X then a lowercase letter matched (X X x).
    #
    # Once we would match the final uppercase (to form X X x Y), that would produce
    # the forbidden pattern; we do NOT count those ways (they go to an "absorbing forbidden" state),
    # so we never add them into our DP. However, from A3[x], seeing an uppercase letter c
    # could also start a new pattern with c as the first letter (i.e. transition to A1[c]),
    # but we do not complete the old pattern (that would be forbidden). Essentially, we "branch":
    # we skip finishing the old pattern and instead start a new one.
    #
    # We process the string from left to right. For each character s (which may be forced uppercase,
    # forced lowercase, or '?'), we compute the new state-values (nA0, nA1, nA2, nA3) from the old
    # (A0, A1, A2, A3).
    #
    # Transitions (for a single forced letter L):
    #
    # 1) "Skip" this character in the subsequence sense:
    #    nA0 += A0
    #    nA1[x] += A1[x]  for all x
    #    nA2[x] += A2[x]  for all x
    #    nA3[x] += A3[x]  for all x
    #
    # 2) "Use" this character in the subsequence sense:
    #    - If L is uppercase c in [0..25]:
    #       from A0 -> A1[c]
    #       from A1[x]:
    #         if x == c -> A2[x]
    #         else       -> A1[c]  (we start a new pattern with c)
    #       from A2[x]: -> A1[c]  (we needed a lowercase to go to A3, so using uppercase
    #                              breaks the old chain; we start a new pattern with c)
    #       from A3[x]: -> A1[c]  (using uppercase here would complete the old pattern => forbidden,
    #                              so we discard that completion path, but can start new with c)
    #
    #    - If L is lowercase c in [0..25]:
    #       from A0 -> A0         (using a lowercase doesn't start the pattern, but we stay in A0)
    #       from A1[x] -> A1[x]   (we're still waiting to match X (2nd uppercase), so no progress)
    #       from A2[x] -> A3[x]   (we needed exactly a lowercase next, so we move to X X x)
    #       from A3[x] -> A3[x]   (already have X X x, we are waiting for final uppercase; using
    #                              a lowercase just keeps us in the same partial state)
    #
    # If s == '?', we multiply the "skip" by 52 (since any of the 52 letters can be skipped),
    # and then do "use" transitions separately for all 26 uppercase letters and 26 lowercase letters.
    #
    # Finally, the answer is A0 + sum(A1[x]) + sum(A2[x]) + sum(A3[x]) after processing all characters.
    #
    # This runs in O(n * 26), which should be acceptable in optimized Python for n up to 3e5.
    #
    # Let's implement it.

    A0 = 1              # dp for "matched nothing yet"
    A1 = [0]*26         # dp for "matched one uppercase letter X"
    A2 = [0]*26         # dp for "matched X X"
    A3 = [0]*26         # dp for "matched X X x"

    for ch in S:
        nA0 = 0
        nA1 = [0]*26
        nA2 = [0]*26
        nA3 = [0]*26

        if ch == '?':
            # Skip transitions (all 52 possibilities are "skipped")
            # Each old state can remain the same in 52 ways
            # A0 -> nA0
            nA0 = (nA0 + A0 * 52) % MOD
            # A1[x] -> nA1[x]
            for x in range(26):
                nA1[x] = (nA1[x] + A1[x] * 52) % MOD
            # A2[x] -> nA2[x]
            for x in range(26):
                nA2[x] = (nA2[x] + A2[x] * 52) % MOD
            # A3[x] -> nA3[x]
            for x in range(26):
                nA3[x] = (nA3[x] + A3[x] * 52) % MOD

            # Now "use" transitions for uppercase letters [0..25]
            # We'll do them in an aggregated way to avoid looping 52 times per old state.
            # First gather sums:
            sumA0 = A0
            sumA1 = sum(A1) % MOD
            sumA2 = sum(A2) % MOD
            sumA3 = sum(A3) % MOD

            # Use uppercase c:
            # from A0 -> A1[c] += A0
            # from A1[x]: if x==c -> A2[x] += A1[x], else -> A1[c] += A1[x]
            # from A2[x]: -> A1[c] += A2[x]
            # from A3[x]: -> A1[c] += A3[x]  (since finishing old pattern is forbidden,
            #                                 we interpret "use" as starting new pattern)
            #
            # We'll handle them in an aggregated manner:
            #  - A0 contributes A0 to each A1[c], total 26*A0 among all c.
            #  - A1[x] contributes A1[x] to A2[x] for exactly 1 matching c,
            #                  and contributes A1[x] to A1[c] for the other 25 c.
            #  - A2[x] contributes A2[x] to A1[c] for all 26 c
            #  - A3[x] contributes A3[x] to A1[c] for all 26 c
            #
            # We'll do it in two steps:
            # Step 1: Add the portion that goes equally to all c in A1, and the portion that goes
            #         to exactly one c in A2.
            # Step 2: Distribute them accordingly.

            # Step 1: from A0 -> 26 * A0 goes into sum of nA1[...] 
            #         from A1[x] -> 1 * A1[x] into nA2[x], plus 25*A1[x] into sum of nA1[!=x]
            #         from A2[x] -> 26*A2[x] into sum of nA1
            #         from A3[x] -> 26*A3[x] into sum of nA1
            # We'll accumulate how much to add to each nA1[x] uniformly (call it addA1All),
            # and also handle the direct nA2[x].

            addA1All = (26 * sumA0) % MOD  # from A0 => 26 ways
            # from A1[x]:
            #   each x => nA2[x] += A1[x] (for matching c = x),
            #             + 25*A1[x] to the sum of nA1[...] (for c != x)
            for x in range(26):
                val = A1[x]
                if val:
                    nA2[x] = (nA2[x] + val) % MOD
                    # 25 * val goes into the "all" bucket for nA1
                    addA1All = (addA1All + val * 25) % MOD

            # from A2[x]: 26 * A2[x] => all nA1
            # from A3[x]: 26 * A3[x] => all nA1
            addA1All = (addA1All + 26 * sumA2 + 26 * sumA3) % MOD

            # Now we distribute addA1All to each nA1[x]
            for x in range(26):
                nA1[x] = (nA1[x] + addA1All) % MOD

            # Now "use" transitions for lowercase letters [0..25].
            # from A0 => remain in A0
            # from A1[x] => remain in A1[x]
            # from A2[x] => go to A3[x]
            # from A3[x] => remain in A3[x]
            #
            # Each of these 4 transitions happens for each of the 26 lowercase letters.
            # So effectively:
            #  - A0 => 26 * A0 goes to nA0
            #  - A1[x] => 26 * A1[x] goes to nA1[x]
            #  - A2[x] => 26 * A2[x] goes to nA3[x]
            #  - A3[x] => 26 * A3[x] goes to nA3[x]
            nA0 = (nA0 + 26 * A0) % MOD
            for x in range(26):
                nA1[x] = (nA1[x] + 26 * A1[x]) % MOD
                nA3[x] = (nA3[x] + 26 * A2[x] + 26 * A3[x]) % MOD

        else:
            # This is a forced single character
            is_upper = ('A' <= ch <= 'Z')
            is_lower = ('a' <= ch <= 'z')

            if is_upper:
                # convert to 0..25
                c = ord(ch) - ord('A')
                # 1) skip transitions
                nA0 = (nA0 + A0) % MOD
                for x in range(26):
                    nA1[x] = (nA1[x] + A1[x]) % MOD
                    nA2[x] = (nA2[x] + A2[x]) % MOD
                    nA3[x] = (nA3[x] + A3[x]) % MOD

                # 2) use transitions
                #    from A0 -> A1[c]
                nA1[c] = (nA1[c] + A0) % MOD

                #    from A1[x]:
                #      if x == c -> A2[x], else -> A1[c]
                for x in range(26):
                    val = A1[x]
                    if val == 0:
                        continue
                    if x == c:
                        nA2[x] = (nA2[x] + val) % MOD
                    else:
                        nA1[c] = (nA1[c] + val) % MOD

                #    from A2[x] -> A1[c]
                for x in range(26):
                    val = A2[x]
                    if val == 0:
                        continue
                    nA1[c] = (nA1[c] + val) % MOD

                #    from A3[x] -> A1[c] (because using uppercase would complete X X x Y => forbidden,
                #                        so we skip completing and treat it as new pattern start)
                for x in range(26):
                    val = A3[x]
                    if val == 0:
                        continue
                    nA1[c] = (nA1[c] + val) % MOD

            elif is_lower:
                # convert to 0..25
                c = ord(ch) - ord('a')
                # 1) skip transitions
                nA0 = (nA0 + A0) % MOD
                for x in range(26):
                    nA1[x] = (nA1[x] + A1[x]) % MOD
                    nA2[x] = (nA2[x] + A2[x]) % MOD
                    nA3[x] = (nA3[x] + A3[x]) % MOD

                # 2) use transitions
                #    from A0 -> A0
                nA0 = (nA0 + A0) % MOD

                #    from A1[x] -> A1[x]
                for x in range(26):
                    val = A1[x]
                    if val:
                        nA1[x] = (nA1[x] + val) % MOD

                #    from A2[x] -> A3[x]
                for x in range(26):
                    val = A2[x]
                    if val:
                        nA3[x] = (nA3[x] + val) % MOD

                #    from A3[x] -> A3[x]
                for x in range(26):
                    val = A3[x]
                    if val:
                        nA3[x] = (nA3[x] + val) % MOD

            else:
                # Invalid input character (should not happen per the problem statement)
                # but just in case, result is 0
                print(0)
                return

        # Move to next
        A0 = nA0 % MOD
        A1 = [v % MOD for v in nA1]
        A2 = [v % MOD for v in nA2]
        A3 = [v % MOD for v in nA3]

    # Final answer is sum of all states that did NOT produce the forbidden subsequence:
    ans = A0
    ans = (ans + sum(A1)) % MOD
    ans = (ans + sum(A2)) % MOD
    ans = (ans + sum(A3)) % MOD

    print(ans % MOD)