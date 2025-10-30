# YOUR CODE HERE
N, Q = map(int, input().split())
x, y = 1, 0
dx, dy = 0, 0
parts = [(1, 0)] * N

def move_head(dx, dy):
    global x, y
    x += dx
    y += dy
    parts[0] = (x, y)

def follow_parts():
    for i in range(1, N):
        parts[i] = parts[i-1]

for _ in range(Q):
    query = list(map(int, input().split()))
    if query[0] == 1:
        if query[1] == 1:
            move_head(0, 1)
        elif query[1] == 2:
            move_head(0, -1)
        elif query[1] == 3:
            move_head(1, 0)
        elif query[1] == 4:
            move_head(-1, 0)
        follow_parts()
    else:
        print(parts[query[1]-1][0], parts[query[1]-1][1])