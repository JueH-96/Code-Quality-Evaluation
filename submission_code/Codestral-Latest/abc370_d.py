# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

index = 0
H = int(data[index])
index += 1
W = int(data[index])
index += 1
Q = int(data[index])
index += 1

walls = [[True] * W for _ in range(H)]
remaining_walls = H * W

for _ in range(Q):
    R_q = int(data[index]) - 1
    index += 1
    C_q = int(data[index]) - 1
    index += 1

    if walls[R_q][C_q]:
        walls[R_q][C_q] = False
        remaining_walls -= 1
    else:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            r, c = R_q + dr, C_q + dc
            while 0 <= r < H and 0 <= c < W:
                if walls[r][c]:
                    walls[r][c] = False
                    remaining_walls -= 1
                    break
                r += dr
                c += dc

print(remaining_walls)