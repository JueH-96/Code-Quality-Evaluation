# Read input
N, M = map(int, input().split())
S = input().strip()
C = list(map(int, input().split()))

# Precompute positions for each color
color_positions = [[] for _ in range(M + 1)]
for idx in range(N):
    color = C[idx]
    color_positions[color].append(idx)

s_list = list(S)

# Process each color in order
for color in range(1, M + 1):
    pos_list = color_positions[color]
    chars = [s_list[p] for p in pos_list]
    if not chars:
        continue
    # Perform right circular shift
    new_chars = [chars[-1]] + chars[:-1]
    for i, p in enumerate(pos_list):
        s_list[p] = new_chars[i]

# Output the result
print(''.join(s_list))