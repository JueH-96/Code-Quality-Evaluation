# YOUR CODE HERE
N = int(input())
P = list(map(int, input().split()))

A = []

for i in range(1, N + 1):
    pos = P[i - 1]  # P_i for number i (convert to 0-indexed)
    # Insert i at position pos (1-indexed)
    # This means insert at index pos-1 (0-indexed)
    A.insert(pos - 1, i)

print(*A)