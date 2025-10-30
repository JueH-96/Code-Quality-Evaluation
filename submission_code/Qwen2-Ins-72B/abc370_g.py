from math import sqrt

MOD = 998244353

def preprocess(N):
    maxp = int(sqrt(N)) + 1
    is_prime = [True] * (maxp + 1)
    for i in range(2, maxp + 1):
        if not is_prime[i]:
            continue
        for j in range(i * 2, maxp + 1, i):
            is_prime[j] = False
    primes = [i for i in range(2, maxp + 1) if is_prime[i]]
    
    cnt = [0] * (maxp + 1)
    for p in primes:
        e = 0
        while p ** (e + 1) <= N:
            e += 1
        for i in range(e, -1, -1):
            cnt[p ** i] += 1
    return cnt

def solve(N, M, cnt):
    dp = [0] * (M + 1)
    dp[0] = 1
    for p in range(2, len(cnt)):
        for i in range(M, 0, -1):
            for j in range(1, cnt[p] + 1):
                if i - j < 0:
                    break
                dp[i] = (dp[i] + dp[i - j]) % MOD
        if (p - 1) % 3 == 0:
            for i in range(M, 0, -1):
                dp[i] = (2 * dp[i] - dp[i - 1]) % MOD
    return dp[M]

def main():
    N, M = map(int, input().split())
    cnt = preprocess(N)
    print(solve(N, M, cnt))

if __name__ == "__main__":
    main()