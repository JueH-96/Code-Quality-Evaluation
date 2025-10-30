N, M = map(int, input().split())
A = list(map(int, input().split()))

# Sort the positions
A.sort()

max_gifts = 0
j = 0

for i in range(N):
    # Move j to include all gifts in [A[i], A[i] + M)
    while j < N and A[j] < A[i] + M:
        j += 1
    # Count gifts in the interval
    max_gifts = max(max_gifts, j - i)

print(max_gifts)