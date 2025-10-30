from collections import Counter

N = int(input())
A = list(map(int, input().split()))

mod = 10**8
A = [(a % mod, i) for i, a in enumerate(A)]
A.sort()

# Count the number of elements that are less than or equal to the current element
cnt = Counter([a for a, _ in A])
total = 0
for i in range(N):
    a, idx = A[i]
    total += a * (i - cnt[a]) * (N - i - 1 + cnt[a])
    cnt[a] -= 1

print(total % mod)