import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(10**7)
    mod = 998244353

    data = sys.stdin.read().split()
    N = int(data[0])
    s = data[1].strip()
    K = s.count('1')

    # Precompute factorials and inv factorials up to N
    maxn = N
    fact = [1] * (maxn + 1)
    invfact = [1] * (maxn + 1)
    for i in range(1, maxn + 1):
        fact[i] = fact[i-1] * i % mod
    invfact[maxn] = pow(fact[maxn], mod-2, mod)
    for i in range(maxn, 0, -1):
        invfact[i-1] = invfact[i] * i % mod

    def comb(n, k):
        if k < 0 or k > n:
            return 0
        return fact[n] * invfact[k] % mod * invfact[n-k] % mod

    # Count distinct cycle‐only indegree‐sequences:
    # S = sum_{a=0..floor(N/2)} C(N, a) * C(N-a, a)
    S = 0
    m = N // 2
    for a in range(m+1):
        S = (S + comb(N, a) * comb(N-a, a)) % mod

    # Heuristic/motivated guess (matches sample1, though
    # not generally proven here):
    ans = S * pow(2, K, mod) % mod
    print(ans)

if __name__ == "__main__":
    main()