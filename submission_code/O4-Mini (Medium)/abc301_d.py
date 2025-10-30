def main():
    import sys
    S = sys.stdin.readline().strip()
    N = int(sys.stdin.readline().strip())
    L = len(S)

    # If N is at least the maximum possible with all '?' = 1, we can short-circuit.
    # Max value for length L is 2^L - 1
    max_all_ones = (1 << L) - 1
    if N >= max_all_ones:
        # Just replace all '?' with '1' and parse.
        ans = 0
        for ch in S:
            ans = (ans << 1) | (1 if ch == '?' else int(ch))
        print(ans)
        return

    # Otherwise, we must enforce <= N.  Represent N in L bits (with leading zeros).
    n_bits = [0] * L
    tmp = N
    # Fill from least significant end
    for i in range(L - 1, -1, -1):
        n_bits[i] = tmp & 1
        tmp >>= 1

    # dp[pos][tight] = can we fill positions pos..L-1 so that the whole number
    # respects the prefix tightness (tight=1 means prefix so far equals N's prefix).
    dp = [[False, False] for _ in range(L + 1)]
    dp[L][0] = dp[L][1] = True  # base case: empty suffix always valid

    for pos in range(L - 1, -1, -1):
        c = S[pos]
        for tight in (0, 1):
            ok = False
            # try both bits if '?', or the fixed bit
            for b in (0, 1):
                if c != '?' and b != (c == '1'):
                    continue
                if tight and b > n_bits[pos]:
                    # would exceed N's prefix
                    continue
                new_tight = tight and (b == n_bits[pos])
                if dp[pos + 1][new_tight]:
                    ok = True
                    break
            dp[pos][tight] = ok

    # If we cannot even start with tight=1, no solution
    if not dp[0][1]:
        print(-1)
        return

    # Reconstruct the maximum valid number
    ans_bits = []
    tight = 1
    for pos in range(L):
        c = S[pos]
        for b in (1, 0):  # try to put 1 first for maximization
            if c != '?' and b != (c == '1'):
                continue
            if tight and b > n_bits[pos]:
                continue
            new_tight = tight and (b == n_bits[pos])
            if dp[pos + 1][new_tight]:
                ans_bits.append(b)
                tight = new_tight
                break

    # Convert bits to integer
    ans = 0
    for b in ans_bits:
        ans = (ans << 1) | b

    print(ans)

if __name__ == "__main__":
    main()