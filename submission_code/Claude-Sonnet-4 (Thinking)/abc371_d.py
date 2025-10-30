import bisect

n = int(input())
X = list(map(int, input().split()))
P = list(map(int, input().split()))

# Compute prefix sums
prefix_sum = [0] * (n + 1)
for i in range(n):
    prefix_sum[i + 1] = prefix_sum[i] + P[i]

q = int(input())
for _ in range(q):
    L, R = map(int, input().split())
    
    # Find the range of villages in [L, R]
    left_idx = bisect.bisect_left(X, L)
    right_idx = bisect.bisect_right(X, R)
    
    # Sum villagers from left_idx to right_idx - 1
    result = prefix_sum[right_idx] - prefix_sum[left_idx]
    print(result)