def main():
    import sys
    data = sys.stdin.read().split()
    if not data: 
        return
    S = data[0].strip()
    mod = 998244353
    # We wish to count the number of ways to assign every '?' a letter from 
    # uppercase+lowercase (52 choices) so that the final string does NOT contain any subsequence that is “DDoS‐type”.
    # A DDoS‐type subsequence is a sequence of four indices i1 < i2 < i3 < i4 for which:
    #   – s[i1] is an uppercase letter,
    #   – s[i2] is the same uppercase letter as s[i1],
    #   – s[i3] is a lowercase letter (any),
    #   – s[i4] is an uppercase letter.
    #
    # One way to count (over all 52^q assignments) the “good” assignments (with no occurrence)
    # is to ‘simulate’ a deterministic automaton that is “aware” of a possible partial match.
    # Because the forbidden pattern is partly “parameterized” by the letter X that appears in positions 1 and 2,
    # we recast our matching of a subsequence in a greedy way. It is a classical fact that a string contains
    # the forbidden subsequence if and only if the following greedy algorithm reaches completion:
    #
    #   Let f = 0.
    #   For each letter x in S (in order):
    #       if f == 0 and x is uppercase:
    #           set candidate = x; f = 1
    #       elif f == 1 and x is uppercase and x == candidate:
    #           f = 2
    #       elif f == 2 and x is lowercase:
    #           f = 3
    #       elif f == 3 and x is uppercase:
    #           f = 4  [found forbidden subsequence]
    #
    # Our DP will “simulate” reading the string from left to right and record the progress f.
    # Because state f = 1,2,3 depends on the candidate letter X chosen at the moment,
    # we maintain arrays dp1, dp2, dp3 indexed by 0...25 corresponding to candidate letter ‘A’..‘Z’.
    #
    # We have four states:
    #   State0: f = 0 (not started).
    #   State1: f = 1 (found an uppercase letter; candidate letter stored as X).
    #   State2: f = 2 (found two occurrences, both equal X).
    #   State3: f = 3 (found X, X and then a lowercase letter).
    # 
    # Note: If we ever take a transition in state3 with an uppercase letter, we would have f=4 meaning forbidden.
    # We do not allow that transition in our count.
    #
    # Transitions when reading a letter c (which might be fixed or if '?' then we sum over possibilities):
    #
    # 1. From state0 (dp0):
    #    – If c is uppercase: then the letter can start a subsequence by setting candidate to c.
    #         Transition: state0 -> state1[c].
    #    – If c is lowercase: no progress; remain in state0.
    #
    # 2. From state1 with candidate X (dp1[X]):
    #    – If c is uppercase:
    #         If c == X: then progress to state2[X] (matched uppercase X a second time).
    #         If c != X: remain in state1[X] (the candidate does not change).
    #    – If c is lowercase: remain in state1[X] (no progress).
    #
    # 3. From state2 with candidate X (dp2[X]):
    #    – If c is lowercase: progress to state3[X].
    #    – If c is uppercase: remain in state2[X].
    #
    # 4. From state3 with candidate X (dp3[X]):
    #    – If c is uppercase: that would complete the forbidden subsequence — we avoid such transitions.
    #    – If c is lowercase: remain in state3[X].
    #
    # We are “reading” the whole assigned string (each letter is determined, even if originally fixed or by a '?').
    # Every letter in the string appears in exactly one way. Thus, our DP on positions will be a
    # recurrence that “processes” the letter (or, in the case of '?', sums over all 52 possibilities)
    # and updates the counts of assignments for each automaton state.
    #
    # We maintain:
    #    dp0: count of assignments (for processed prefix) ending in state0.
    #    dp1: 26-element list for state1 (for each candidate letter).
    #    dp2: similarly for state2.
    #    dp3: similarly for state3.
    #
    # Our final answer is the sum of counts in dp0, dp1, dp2, and dp3 (all non-forbidden outcomes).
    
    dp0 = 1
    dp1 = [0]*26
    dp2 = [0]*26
    dp3 = [0]*26

    for ch in S:
        ndp0 = 0
        ndp1 = [0]*26
        ndp2 = [0]*26
        ndp3 = [0]*26
        if ch != '?':
            # Fixed letter.
            if ch.isupper():
                # From state0: if the letter is uppercase, we start a subsequence.
                idx = ord(ch) - ord('A')
                ndp1[idx] = (ndp1[idx] + dp0) % mod
            else:
                # lowercase: state0 remains.
                ndp0 = (ndp0 + dp0) % mod

            # Process dp1: state1 with candidate X.
            if ch.isupper():
                idx = ord(ch) - ord('A')
                for X in range(26):
                    if dp1[X]:
                        if X == idx:
                            # c equals candidate => progress to state2.
                            ndp2[X] = (ndp2[X] + dp1[X]) % mod
                        else:
                            # Otherwise, no progress.
                            ndp1[X] = (ndp1[X] + dp1[X]) % mod
            else:
                # c is lowercase: remain in state1.
                for X in range(26):
                    if dp1[X]:
                        ndp1[X] = (ndp1[X] + dp1[X]) % mod

            # Process dp2: state2 with candidate X.
            if ch.isupper():
                # Uppercase: stay in state2.
                for X in range(26):
                    if dp2[X]:
                        ndp2[X] = (ndp2[X] + dp2[X]) % mod
            else:
                # Lowercase: transition to state3.
                for X in range(26):
                    if dp2[X]:
                        ndp3[X] = (ndp3[X] + dp2[X]) % mod

            # Process dp3: state3 with candidate X.
            if ch.isupper():
                # Uppercase would complete a forbidden subsequence.
                # So we add nothing.
                pass
            else:
                # Lowercase: remain in state3.
                for X in range(26):
                    if dp3[X]:
                        ndp3[X] = (ndp3[X] + dp3[X]) % mod

        else:
            # ch == '?' so consider all 52 possibilities.
            # Possibilities: 26 lowercase, 26 uppercase.
            #
            # Process dp0:
            #   For lowercase letter: remain in state0. (26 ways)
            ndp0 = (ndp0 + dp0 * 26) % mod
            #   For uppercase letter: transition to state1, candidate = that letter.
            for X in range(26):
                ndp1[X] = (ndp1[X] + dp0) % mod

            # Process dp1: for each candidate letter X.
            for X in range(26):
                if dp1[X]:
                    # When the chosen letter is uppercase:
                    #   If the chosen letter equals candidate X (1 possibility) -> state2.
                    ndp2[X] = (ndp2[X] + dp1[X]) % mod
                    #   If the chosen letter is uppercase but not equal (25 possibilities) -> remain in state1.
                    ndp1[X] = (ndp1[X] + dp1[X] * 25) % mod
                    # When the letter is lowercase (26 possibilities) -> remain in state1.
                    ndp1[X] = (ndp1[X] + dp1[X] * 26) % mod

            # Process dp2: state2 for candidate X.
            for X in range(26):
                if dp2[X]:
                    # Uppercase letter (26 ways): remain in state2.
                    ndp2[X] = (ndp2[X] + dp2[X] * 26) % mod
                    # Lowercase letter (26 ways): move to state3.
                    ndp3[X] = (ndp3[X] + dp2[X] * 26) % mod

            # Process dp3: state3 for candidate X.
            for X in range(26):
                if dp3[X]:
                    # Uppercase letter would complete the forbidden pattern: 0 contribution.
                    # Lowercase letter (26 ways): remain in state3.
                    ndp3[X] = (ndp3[X] + dp3[X] * 26) % mod

        dp0, dp1, dp2, dp3 = ndp0, ndp1, ndp2, ndp3

    # The final answer is the sum over states that have not reached the forbidden full match.
    ans = dp0
    ans = (ans + sum(dp1)) % mod
    ans = (ans + sum(dp2)) % mod
    ans = (ans + sum(dp3)) % mod
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()