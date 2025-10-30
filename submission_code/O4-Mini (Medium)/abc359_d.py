def main():
    import sys
    input = sys.stdin.readline
    MOD = 998244353

    N, K = map(int, input().split())
    S = input().strip()

    # Count of '?' for trivial power if K > N
    # But our DP handles K > N naturally.

    # Precompute palindrome flags for all K-bit numbers
    # pal[w] = True if the K-bit word w is a palindrome
    if K <= 20:
        max_w = 1 << K
        pal = [False] * max_w
        for w in range(max_w):
            ok = True
            for j in range(K // 2):
                if ((w >> j) & 1) != ((w >> (K - 1 - j)) & 1):
                    ok = False
                    break
            if ok:
                pal[w] = True
    else:
        # K up to 10 by constraints
        pal = []

    # DP over positions, remembering last K-1 bits in reversed order
    # mask bit j = b_{i-1-j}, for j in [0..K-2]
    M1 = 1 << max(0, K - 1)
    dp_old = [0] * M1
    dp_old[0] = 1

    for i, ch in enumerate(S):
        dp_new = [0] * M1
        # Determine possible bits at position i
        if ch == 'A':
            choices = (0,)
        elif ch == 'B':
            choices = (1,)
        else:
            choices = (0, 1)

        # For each mask and each choice
        for mask in range(M1):
            v = dp_old[mask]
            if v == 0:
                continue
            # Try to place bit c
            for c in choices:
                w = ((mask << 1) | c)
                # If we've reached at least K chars (i >= K-1), check the K-window
                if i >= K - 1:
                    # w now has K bits (lowest K bits), check palindrome
                    if pal[w & ((1 << K) - 1)]:
                        continue
                # New mask is keep only last K-1 bits of w
                new_mask = w & (M1 - 1)
                dp_new[new_mask] = (dp_new[new_mask] + v) % MOD

        dp_old = dp_new

    # Sum all final states
    ans = sum(dp_old) % MOD
    print(ans)

if __name__ == "__main__":
    main()