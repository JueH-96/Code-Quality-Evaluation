# Read input
N = int(input())
P = list(map(int, input().split()))
Q = list(map(int, input().split()))

# Create mapping from bib number to person position (1-based)
bib_to_pos = {}
for i in range(N):
    bib_to_pos[Q[i]] = i + 1

# For each bib number i, find who they're staring at
result = []
for i in range(1, N + 1):
    # Find position of person wearing bib i
    pos = bib_to_pos[i]
    # Find who they're staring at (P[pos-1])
    staring_at_pos = P[pos-1]
    # Get bib number of that person (Q[staring_at_pos-1])
    result.append(Q[staring_at_pos-1])

# Print result
print(" ".join(map(str, result)))