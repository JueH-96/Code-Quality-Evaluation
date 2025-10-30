N, M = map(int, input().split())
A = list(map(int, input().split()))

# Calculate cumulative sums
cum_sum = [0] * (N+1)
for i in range(N):
    cum_sum[i+1] = cum_sum[i] + A[i]

# For each pair of rest areas, check if distance is multiple of M
count = 0
for s in range(N):
    for t in range(N):
        if s == t:
            continue
            
        # Get clockwise distance from s to t
        if s < t:
            dist = cum_sum[t] - cum_sum[s]
        else:
            dist = cum_sum[N] - cum_sum[s] + cum_sum[t]
            
        if dist % M == 0:
            count += 1

print(count)