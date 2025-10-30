MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    constraints = [[] for _ in range(N+1)]  # constraints[X] is list of (L, R)
    for _ in range(M):
        L = int(input[ptr])
        ptr += 1
        R = int(input[ptr])
        ptr += 1
        X = int(input[ptr])
        ptr += 1
        constraints[X].append( (L, R) )

    # Preprocess bad array
    size = N + 2
    bad = [ [ [False]*(size) for _ in range(size) ] for _ in range(size) ]

    for X in range(1, N+1):
        diff = [ [0]*size for _ in range(size) ]
        for (L, R) in constraints[X]:
            # Apply four-point update
            diff[1][R] += 1
            diff[1][N+1] -= 1
            diff[L+1][R] -= 1
            diff[L+1][N+1] += 1

        # Compute row-wise prefix sums
        for i in range(1, N+1):
            for j in range(1, N+1):
                diff[i][j] += diff[i][j-1]

        # Compute column-wise prefix sums
        for j in range(1, N+1):
            for i in range(1, N+1):
                diff[i][j] += diff[i-1][j]

        # Fill bad[X][l][r]
        for l in range(1, N+1):
            for r in range(1, N+1):
                if diff[l][r] >= 1:
                    bad[X][l][r] = True
                else:
                    bad[X][l][r] = False

    # DP table
    dp = [ [0]*(N+2) for _ in range(N+2) ]

    # Base case: length 1
    for i in range(1, N+1):
        if bad[i][i][i]:
            dp[i][i] = 0
        else:
            dp[i][i] = 1

    # Fill DP for longer lengths
    for length in range(2, N+1):
        for l in range(1, N - length + 2):
            r = l + length - 1
            total = 0
            for m in range(l, r+1):
                if not bad[m][l][r]:
                    total = (total + dp[l][m-1] * dp[m+1][r]) % MOD
            dp[l][r] = total % MOD

    print(dp[1][N] % MOD)

if __name__ == "__main__":
    main()