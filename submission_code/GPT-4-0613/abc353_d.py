MOD = 998244353
N = int(input())
A = list(map(int, input().split()))
S = [0]*11
ans = 0
for a in A:
    k = len(str(a))
    t = a
    for _ in range(k):
        ans += S[t%10]*pow(10, _, MOD)
        ans %= MOD
        t //= 10
    t = a
    for _ in range(k):
        S[t%10] += pow(10, _, MOD)
        S[t%10] %= MOD
        t //= 10
print(ans)