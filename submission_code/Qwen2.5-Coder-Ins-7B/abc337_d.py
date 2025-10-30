import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
K = int(data[2])
grid = data[3:]

def check_horizontal():
    for i in range(H):
        count = 0
        for j in range(W):
            if grid[i][j] == '.':
                count += 1
            elif grid[i][j] == 'o':
                if count >= K:
                    return count
                count = 0
        if count >= K:
            return count
    return -1

def check_vertical():
    for j in range(W):
        count = 0
        for i in range(H):
            if grid[i][j] == '.':
                count += 1
            elif grid[i][j] == 'o':
                if count >= K:
                    return count
                count = 0
        if count >= K:
            return count
    return -1

horizontal = check_horizontal()
vertical = check_vertical()

if horizontal == -1 and vertical == -1:
    print(-1)
else:
    print(min(horizontal, vertical))