def main():
    import sys
    sys.setrecursionlimit(10**7)

    MOD = 998244353

    S = sys.stdin.readline().rstrip('
')
    n = len(S)

    # Count how many '?' in total
    q = S.count('?')

    # Precompute 52^q mod
    pow_52_q = pow(52, q, MOD)

    # We'll implement a DP with the following states (all mod 998244353):
    # dp0: number of ways so far (filling up to current index) that have NOT started matching
    # dp1[x]: ways that have matched exactly 1 uppercase letter "X" (0<=x<26)
    # dp2[x]: ways that have matched "X X" (two uppercase letters, same X)
    # dp3[x]: ways that have matched "X X y" (that is uppercase X, uppercase X, lowercase y)
    # dp4: ways that have formed the full subsequence "X X y Z" at least once.

    # Initialize
    dp0 = 1
    dp1 = [0]*26
    dp2 = [0]*26
    dp3 = [0]*26
    dp4 = 0

    # A small helper: map characters to forced uppercase/lowercase indices
    # If c in A..Z, forcedU = ord(c) - ord('A'), forcedL = -1
    # If c in a..z, forcedL = ord(c) - ord('a'), forcedU = -1
    # If c=='?', forcedU = forcedL = -2 (meaning we can choose)
    # We'll also define how many ways c can be assigned to any letter (anyW),
    # how many ways c can be uppercase (uW), and how many ways c can be lowercase (lW),
    # when we want to "use" c for that role in the subsequence.

    for c in S:
        if c == '?':
            forcedU = -2  # can be any uppercase
            forcedL = -2  # can be any lowercase
            anyW = 52
            uW = 26
            lW = 26
        elif 'A' <= c <= 'Z':
            forcedU = ord(c) - ord('A')
            forcedL = -1
            anyW = 1
            uW = 1  # exactly one way to be uppercase (that letter)
            lW = 0
        else:
            # c in 'a'..'z'
            forcedU = -1
            forcedL = ord(c) - ord('a')
            anyW = 1
            uW = 0
            lW = 1  # exactly one way to be that lowercase letter

        # Prepare next states
        next_dp0 = 0
        next_dp1 = [0]*26
        next_dp2 = [0]*26
        next_dp3 = [0]*26
        next_dp4 = 0

        # 1) Skip transitions (not using this character in the subsequence):
        # dp0 -> dp0
        next_dp0 = (dp0 * anyW) % MOD
        # dp4 -> dp4 (once pattern formed, it stays formed no matter how we fill the rest)
        next_dp4 = (dp4 * anyW) % MOD
        # dp1[x] -> dp1[x]
        for x in range(26):
            if dp1[x] != 0:
                next_dp1[x] = (next_dp1[x] + dp1[x]*anyW) % MOD
        # dp2[x] -> dp2[x]
        for x in range(26):
            if dp2[x] != 0:
                next_dp2[x] = (next_dp2[x] + dp2[x]*anyW) % MOD
        # dp3[x] -> dp3[x]
        for x in range(26):
            if dp3[x] != 0:
                next_dp3[x] = (next_dp3[x] + dp3[x]*anyW) % MOD

        # 2) Use transitions
        # dp0 -> dp1[x], if c can be uppercase letter x
        if forcedU == -2:
            # c=='?' => can be any uppercase letter x
            # for each x, next_dp1[x] += dp0
            if dp0 != 0:
                val = dp0  # for each x
                for x in range(26):
                    next_dp1[x] = (next_dp1[x] + val) % MOD
        elif forcedU >= 0:
            # c is forced uppercase letter forcedU
            x = forcedU
            next_dp1[x] = (next_dp1[x] + dp0) % MOD

        # dp1[x] -> dp2[x], if c can be uppercase letter x
        if forcedU == -2:
            # c=='?' => can be any uppercase letter
            # for x in [0..25], next_dp2[x] += dp1[x]
            for x in range(26):
                if dp1[x] != 0:
                    next_dp2[x] = (next_dp2[x] + dp1[x]) % MOD
        elif forcedU >= 0:
            xU = forcedU
            if dp1[xU] != 0:
                next_dp2[xU] = (next_dp2[xU] + dp1[xU]) % MOD

        # dp2[x] -> dp3[x], if c can be a lowercase letter
        if lW > 0:
            # If c can be *some* lowercase letter. 
            # We have to multiply each dp2[x] by the number of ways
            # this character can be chosen as a single lowercase letter.
            # If forcedL >= 0, that's 1 way. If c=='?', that's 26 ways.
            for x in range(26):
                if dp2[x] != 0:
                    next_dp3[x] = (next_dp3[x] + dp2[x]*lW) % MOD

        # dp3[x] -> dp4, if c can be an uppercase letter (the final 'Z' in the pattern)
        # we add dp3[x]*uW to next_dp4, for each x
        if uW > 0:
            sum_dp3 = 0
            for x in range(26):
                sum_dp3 = (sum_dp3 + dp3[x]) % MOD
            # Each dp3[x] can complete the pattern with an uppercase letter in uW ways
            # So add sum_dp3 * uW to next_dp4
            next_dp4 = (next_dp4 + sum_dp3*uW) % MOD

        # Update dp for next iteration
        dp0 = next_dp0
        dp1 = next_dp1
        dp2 = next_dp2
        dp3 = next_dp3
        dp4 = next_dp4

    # dp4 now holds the number of ways (out of all possible fillings) that contain
    # at least one subsequence of the form "X X y Z" (DDoS-type)
    # The total number of ways to fill the string is 52^q
    # The answer is "total ways" - "ways that contain a DDoS-type subsequence"
    ans = pow_52_q - dp4
    ans %= MOD

    print(ans)

# Do not forget to call main()
if __name__ == "__main__":
    main()