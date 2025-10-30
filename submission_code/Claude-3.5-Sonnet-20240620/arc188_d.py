# YOUR CODE HERE
MOD = 998244353

def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result = (result * i) % MOD
    return result

def inverse(x):
    return pow(x, MOD - 2, MOD)

def solve(N, A, B):
    used = [False] * (2 * N + 1)
    for a in A:
        used[a] = True
    for b in B:
        if b != -1:
            used[b] = True

    free_positions = [i for i in range(N) if B[i] == -1]
    free_numbers = [i for i in range(1, 2 * N + 1) if not used[i]]
    M = len(free_positions)

    dp = [0] * (M + 1)
    dp[0] = 1

    for i in range(M):
        new_dp = [0] * (M + 1)
        for j in range(M + 1):
            if dp[j] == 0:
                continue
            if j < M:
                new_dp[j + 1] = (new_dp[j + 1] + dp[j]) % MOD
            if i < j:
                new_dp[j] = (new_dp[j] + dp[j] * (j - i)) % MOD
        dp = new_dp

    result = dp[M]
    result = (result * factorial(N)) % MOD
    for i in range(3):
        used = [False] * (N + 1)
        for j in range(N):
            if i == 0 and A[j] <= N:
                used[A[j]] = True
            elif i == 1 and A[j] > N:
                used[A[j] - N] = True
            elif i == 2 and B[j] != -1:
                if B[j] <= N:
                    used[B[j]] = True
                else:
                    used[B[j] - N] = True
        free = N - sum(used)
        result = (result * inverse(factorial(free))) % MOD

    return result

N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

print(solve(N, A, B))