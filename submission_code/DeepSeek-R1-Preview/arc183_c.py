MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    forbidden = [[] for _ in range(N+1)]  # 1-based

    for _ in range(M):
        L = int(input[ptr])
        ptr += 1
        R = int(input[ptr])
        ptr += 1
        X = int(input[ptr])
        ptr += 1
        forbidden[X].append((L, R))

    # Precompute for each x, the intervals where x is forbidden, and precompute their masks
    x_masks = [[] for _ in range(N+1)]
    for x in range(1, N+1):
        for (L, R) in forbidden[x]:
            mask = 0
            for i in range(L, R+1):
                if i != x:
                    mask |= 1 << (i-1)
            x_masks[x].append(mask)

    # DP: dp[mask] = number of ways to reach this mask
    dp = [0] * (1 << N)
    dp[0] = 1

    for mask in range(0, 1 << N):
        if dp[mask] == 0:
            continue
        for x in range(1, N+1):
            x_bit = 1 << (x-1)
            if (mask & x_bit) == 0:
                # Check all intervals for x
                valid = True
                for m in x_masks[x]:
                    if (mask & m) == 0:
                        valid = False
                        break
                if valid:
                    new_mask = mask | x_bit
                    dp[new_mask] = (dp[new_mask] + dp[mask]) % MOD

    print(dp[(1 << N) - 1] % MOD)

if __name__ == '__main__':
    main()