# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
Q = int(data[2])

walls = [[True] * W for _ in range(H)]

index = 3
for _ in range(Q):
    R = int(data[index]) - 1
    C = int(data[index + 1]) - 1
    index += 2
    
    if walls[R][C]:
        walls[R][C] = False
    else:
        for i in range(R - 1, -1, -1):
            if walls[i][C]:
                walls[i][C] = False
                break
        for i in range(R + 1, H):
            if walls[i][C]:
                walls[i][C] = False
                break
        for j in range(C - 1, -1, -1):
            if walls[R][j]:
                walls[R][j] = False
                break
        for j in range(C + 1, W):
            if walls[R][j]:
                walls[R][j] = False
                break

remaining_walls = sum(row.count(True) for row in walls)
print(remaining_walls)