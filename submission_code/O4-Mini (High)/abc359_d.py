import sys

def main():
    input = sys.stdin.readline
    MOD = 998244353

    N, K = map(int, input().split())
    S = input().strip()

    # Precompute which K-bit patterns are palindromes
    maxKmask = 1 << K
    is_pal = [False] * maxKmask
    for m in range(maxKmask):
        ok = True
        # check bits m[0..K-1] vs m[K-1..0]
        for i in range(K):
            if ((m >> i) & 1) != ((m >> (K - 1 - i)) & 1):
                ok = False
                break
        is_pal[m] = ok

    # We'll keep a DP over the last K-1 bits.
    # dp[mask] = number of ways so far where the last (up to) K-1 bits
    # are equal to `mask` (mask is in [0..2^(K-1)-1]).
    M = 1 << (K - 1)
    dp = [0] * M
    dp[0] = 1

    full_mask_bits = (1 << K) - 1
    last_Kminus1_mask = (1 << (K - 1)) - 1

    for i, ch in enumerate(S):
        dp2 = [0] * M

        # Determine which bit(s) we can place here
        if ch == 'A':
            choices = (0,)
        elif ch == 'B':
            choices = (1,)
        else:  # ch == '?'
            choices = (0, 1)

        for mask in range(M):
            ways = dp[mask]
            if ways == 0:
                continue
            for bit in choices:
                # form the new K-bit window
                fm = ((mask << 1) | bit) & full_mask_bits
                # if we have built >= K characters, we must avoid palindromes of length K
                if i + 1 >= K and is_pal[fm]:
                    continue
                # new state is the lower K-1 bits of fm
                nm = fm & last_Kminus1_mask
                dp2[nm] = (dp2[nm] + ways) % MOD

        dp = dp2

    answer = sum(dp) % MOD
    print(answer)


if __name__ == "__main__":
    main()