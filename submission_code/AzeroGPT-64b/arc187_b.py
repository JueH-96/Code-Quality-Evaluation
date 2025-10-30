from itertools import accumulate

N, M, *B = map(int, open(0).read().split())

DSU = [None] * (N + 1)
class DisjointSet:  # O(α(n))
    def __init__(self, size):
        for i in range(size):
            DSU[i] = i

    def root(self, i):
        if DSU[i] != i:
            DSU[i] = self.root(DSU[i])
        return DSU[i]

    def number(self, i):
        return self.root(i)

    def unite(self, i, j):  # O(α(n))
        i, j = self.root(i), self.root(j)
        if i == j:
            return
        DSU[i] = j

    def same(self, i, j):
        return self.root(i) == self.root(j)

dp = [0] * (N + 1)
dp[0] = 1
dp[1] = 0
for i in range(2, N + 1):
    dp[i] = (1 + i - 1) * dp[i - 1] + (i - 1) * dp[i - 2]
dp = list(accumulate(dp))[-1]

ii = [-1] + list(range(N)) + [N]
prev = [None] * N
for i, b in enumerate(B):
    if prev[b - 1] is not None:
        ii[i], ii[prev[b - 1]] = ii[prev[b - 1]], ii[i]
    prev[b - 1] = i

dsuR, dsuL = DisjointSet(N + 1), DisjointSet(N + 1)
q = B.count(-1)
for i in ii[1:N + 1]:
    if B[i] == -1:
        continue
    dsuR.unite(i, ii[B[i] - 1])
    dsuL.unite(ii[B[i] + 1], i)

components = [0] * (N + 1)
for i in range(1, N + 1):
    components[dsuR.root(i)] += 1

dp = dp % 998244353
result = dp * pow(M, q - 1, 998244353)

for lr, lo, rr, ro in zip(components, accumulate(components), reversed(components), accumulate(reversed(components))):
    li, ri = dsuL.number(dsuL.root(lo + 1)), dsuR.number(dsuR.root(N - ro - 1))
    result = (result \
              + components[li] * components[ri] * pow(M, q, 998244353) \
              + components[li] * (components[li] - 1) // 2 * pow(M, q, 998244353) \
              + components[ri] * (components[ri] - 1) // 2 * pow(M, q, 998244353) \
              - 1) % 998244353

for i in range(N):
    if B[i] != -1:
        continue
    li, ri = dsuL.number(dsuL.root(i + 1)), dsuR.number(dsuR.root(N - i - 1))
    ll, rl = components[li], components[ri]
    lr, llr, lrr, lrly, lrrt = ll + rl, ll - components[dsuL.root(i)], rl - components[dsuR.root(i)], ll - llr, rl - lrr
    result = (result \
              + (llr * lrr * M + 2 * llr + 2 * lrr - 4 + 2 * pow(M, q, 998244353) - pow(M, q - 1, 998244353)) \
              - (ll - lrly + lrly * M) * (rl - lrrt + lrrt * M) \
              - ((llr - lrly) * M + lrly) * ((lrr - lrrt) * M + lrrt)) % 998244353

    lroc = components[dsuL.root(i)]
    droc = components[dsuR.root(i)]
    dsc = dsuL.root(i + lroc), dsuR.root(i + droc)

    if droc > 0:
        dsuR.unite(i, dsuR.root(i + droc))
        components[dsuR.root(i)] += droc
    if lroc > 0:
        dsuL.unite(i - lroc, i)
        components[dsuL.root(i)] += lroc

    for l, r in zip(dsc, dsc[::-1]):
        ll, rl = components[l] - components[dsuL.root(i)], components[r] - components[dsuR.root(i)]
        result = (result - ll * rl * M % 998244353) % 998244353

print(result)