from collections import defaultdict

N, M = map(int, input().split())
S = input()
C = list(map(int, input().split()))

# Create a dictionary to store the indices of each color
indices = defaultdict(list)
for i, color in enumerate(C):
    indices[color].append(i)

# Perform the operations for each color
for color in range(1, M + 1):
    idx_list = indices[color]
    k = len(idx_list)
    if k == 1:
        continue

    # Find the starting index of the cycle
    start_idx = idx_list[0]
    i = start_idx
    cycle = [i]
    while True:
        i = (i + 1) % k
        if i == start_idx:
            break
        cycle.append(idx_list[i])

    # Perform the right circular shift on the cycle
    shift = 1
    for i in range(len(cycle)):
        S = S[:cycle[i]] + S[cycle[(i + shift) % len(cycle)]] + S[cycle[i] + 1:]

print(S)