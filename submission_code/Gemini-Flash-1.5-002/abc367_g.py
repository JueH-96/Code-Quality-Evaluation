# YOUR CODE HERE
N, M, K = map(int, input().split())
A = list(map(int, input().split()))
MOD = 998244353

ans = 0
for i in range(1, 1 << N):
    subsequence = []
    for j in range(N):
        if (i >> j) & 1:
            subsequence.append(A[j])
    xor_sum = 0
    for x in subsequence:
        xor_sum ^= x
    if len(subsequence) % M == 0:
        ans = (ans + pow(xor_sum, K, MOD)) % MOD

print(ans)