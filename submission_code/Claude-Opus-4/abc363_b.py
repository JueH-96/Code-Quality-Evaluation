# YOUR CODE HERE
N, T, P = map(int, input().split())
L = list(map(int, input().split()))

# Calculate days needed for each person to reach hair length T
days_needed = []
for i in range(N):
    if L[i] >= T:
        days_needed.append(0)
    else:
        days_needed.append(T - L[i])

# Sort the days needed
days_needed.sort()

# The P-th person (1-indexed) corresponds to index P-1 (0-indexed)
print(days_needed[P-1])