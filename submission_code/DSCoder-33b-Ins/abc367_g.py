# YOUR CODE HERE

MOD = 998244353

def solve():
    N, M, K = map(int, input().split())
    A = list(map(int, input().split()))

    # Precompute powers of 2
    pow2 = [1]
    for _ in range(N):
        pow2.append(pow2[-1] * 2 % MOD)

    # Precompute powers of K
    powK = [1]
    for _ in range(N):
        powK.append(powK[-1] * K % MOD)

    # Precompute XORs
    xor = [[0] * (1 << N) for _ in range(N + 1)]
    for i in range(N):
        xor[i + 1][1 << i] = A[i]
        for S in range(1 << i):
            xor[i + 1][S | (1 << i)] = xor[i][S] ^ A[i]

    # DP
    dp = [0] * (1 << N)
    dp[0] = 1
    for S in range(1, 1 << N):
        cnt = bin(S).count('1')
        for i in range(N):
            if (S >> i) & 1:
                dp[S] = (dp[S] + dp[S ^ (1 << i)] * pow2[cnt - 1] % MOD) % MOD

    # Calculate answer
    ans = 0
    for S in range(1, 1 << N):
        cnt = bin(S).count('1')
        if cnt % M == 0:
            ans = (ans + xor[cnt][S] * powK[cnt] % MOD * dp[((1 << N) - 1) ^ S] % MOD) % MOD

    print(ans)

solve()