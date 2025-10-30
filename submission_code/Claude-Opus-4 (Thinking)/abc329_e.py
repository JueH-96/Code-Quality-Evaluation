# YOUR CODE HERE
N, M = map(int, input().split())
S = input().strip()
T = input().strip()

# Find all valid positions where we can place T
valid_positions = []
for i in range(N - M + 1):
    if S[i:i+M] == T:
        valid_positions.append(i)

# Check if every position can be covered by a valid placement
possible = True
for i in range(N):
    covered = False
    for j in valid_positions:
        if j <= i < j + M:
            covered = True
            break
    if not covered:
        possible = False
        break

print("Yes" if possible else "No")