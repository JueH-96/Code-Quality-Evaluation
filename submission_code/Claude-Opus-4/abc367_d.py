# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Calculate prefix sums
prefix = [0]
for i in range(N):
    prefix.append(prefix[-1] + A[i])

total_sum = prefix[N]
count = 0

# Check all pairs (s, t) where s != t
for s in range(1, N + 1):
    for t in range(1, N + 1):
        if s == t:
            continue
        
        # Calculate clockwise distance from s to t
        if s < t:
            # Direct path from s to t
            distance = prefix[t - 1] - prefix[s - 1]
        else:
            # Go around: from s to N, then from 1 to t
            distance = (prefix[N] - prefix[s - 1]) + prefix[t - 1]
        
        if distance % M == 0:
            count += 1

print(count)