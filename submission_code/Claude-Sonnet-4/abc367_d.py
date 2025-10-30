# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Calculate prefix sums
prefix = [0] * (N + 2)
for i in range(1, N + 1):
    prefix[i] = prefix[i-1] + A[i-1]
prefix[N+1] = prefix[N]  # Total distance around the lake

count = 0

# Check all pairs (s, t) where s != t
for s in range(1, N + 1):
    for t in range(1, N + 1):
        if s != t:
            if s < t:
                distance = prefix[t] - prefix[s]
            else:  # s > t
                distance = prefix[N+1] - prefix[s] + prefix[t]
            
            if distance % M == 0:
                count += 1

print(count)