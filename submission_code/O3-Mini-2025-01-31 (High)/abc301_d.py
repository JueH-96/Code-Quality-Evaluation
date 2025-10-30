def main():
    import sys
    sys.setrecursionlimit(10000)
    
    data = sys.stdin.read().split()
    if not data:
        return
    S = data[0].strip()
    N = int(data[1])
    
    L = len(S)
    # Get the binary representation of N (without the '0b' prefix)
    nbin = bin(N)[2:]
    L_N = len(nbin)
    
    # If S has fewer digits than the bit-length of N, then every number
    # represented by S (which is at most 2^L - 1) is <= N.
    # So the best candidate is obtained by replacing every '?' with '1'.
    if L < L_N:
        max_candidate_str = ''.join('1' if ch == '?' else ch for ch in S)
        print(int(max_candidate_str, 2))
        return

    # Otherwise, view S as an L-digit binary number (leading zeros allowed)
    # and compare it directly to N. If L > L_N, then pad N's binary with
    # leading zeros to get an L-digit number.
    padded_N = nbin if L == L_N else nbin.rjust(L, '0')
    
    # We use DP to decide, digit by digit, which choices for '?' yield
    # a valid binary number <= N while maximizing the result.
    # State: dp(i, less) where i is the current index (0-indexed, most significant)
    # and less is a boolean indicating if the prefix we have so far is already
    # strictly less than padded_N's corresponding prefix.
    
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def dp(i, less):
        # When no more digits, the contribution is 0.
        if i == L:
            return 0
        
        best = None
        # Weight (power of 2) of the i-th digit:
        weight = 1 << (L - i - 1)
        current_n_digit = int(padded_N[i])
        ch = S[i]
        if ch in "01":
            d = int(ch)
            if not less:
                # Must not exceed the corresponding digit of padded_N.
                if d > current_n_digit:
                    return None
                # If d is strictly less, then our future digits have no restrictions.
                new_less = (d < current_n_digit)
            else:
                new_less = True
            sub = dp(i + 1, new_less)
            if sub is None:
                return None
            best = d * weight + sub
        else:
            # When S[i] is '?', try both choices, trying 1 first to maximize.
            for d in [1, 0]:
                if not less and d > current_n_digit:
                    continue  # This choice would exceed N's digit.
                new_less = (not less and d < current_n_digit) or less
                sub = dp(i + 1, new_less)
                if sub is not None:
                    candidate = d * weight + sub
                    if best is None or candidate > best:
                        best = candidate
        return best

    ans = dp(0, False)
    if ans is None:
        print(-1)
    else:
        print(ans)

if __name__ == '__main__':
    main()