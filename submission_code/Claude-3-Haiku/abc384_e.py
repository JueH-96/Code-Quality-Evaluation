# Read the input
H, W, X = map(int, input().split())
P, Q = map(int, input().split())
S = []
for _ in range(H):
    S.append(list(map(int, input().split())))

# Initialize Takahashi's strength
strength = S[P-1][Q-1]

# Perform the absorption actions
while True:
    max_strength = 0
    max_i, max_j = -1, -1
    for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        i, j = P-1 + di, Q-1 + dj
        if 0 <= i < H and 0 <= j < W and S[i][j] < strength / X:
            if S[i][j] > max_strength:
                max_strength = S[i][j]
                max_i, max_j = i, j
    if max_i == -1:
        break
    strength += max_strength
    S[max_i][max_j] = 0

# Print the result
print(strength)