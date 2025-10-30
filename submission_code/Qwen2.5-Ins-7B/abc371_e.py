# YOUR CODE HERE
from collections import defaultdict

N = int(input())
A = list(map(int, input().split()))

ans = 0
last_seen = defaultdict(int)
for i in range(N):
    seen = set()
    for j in range(i, N):
        seen.add(A[j])
        ans += len(seen) - (len(seen) - 1) * (j - i + 1 - (A[j] in last_seen))
        last_seen[A[j]] = j

print(ans)