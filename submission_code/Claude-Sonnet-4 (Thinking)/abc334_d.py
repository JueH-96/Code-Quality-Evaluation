import bisect

N, Q = map(int, input().split())
R = list(map(int, input().split()))

# Sort reindeer requirements
R.sort()

# Compute prefix sums
prefix = [0]
for r in R:
    prefix.append(prefix[-1] + r)

for _ in range(Q):
    X = int(input())
    # Find the largest k such that prefix[k] <= X
    k = bisect.bisect_right(prefix, X) - 1
    print(k)