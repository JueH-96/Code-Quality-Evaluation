import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
P_coords = []
for _ in range(N):
    x = int(data[index])
    y = int(data[index + 1])
    index += 2
    P_coords.append((x, y))
Q_coords = []
for _ in range(N):
    cx = int(data[index])
    cy = int(data[index + 1])
    index += 2
    Q_coords.append((cx, cy))
P_indices_sorted = sorted(range(N), key=lambda i: (P_coords[i][0], P_coords[i][1]))
Q_indices_sorted = sorted(range(N), key=lambda i: (Q_coords[i][0], Q_coords[i][1]))
match_P_to_Q = [0] * N
for k in range(N):
    p_idx = P_indices_sorted[k]
    q_idx = Q_indices_sorted[k]
    match_P_to_Q[p_idx] = q_idx
R_list = [q_idx + 1 for q_idx in match_P_to_Q]
print(' '.join(map(str, R_list)))