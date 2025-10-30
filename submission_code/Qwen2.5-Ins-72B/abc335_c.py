# YOUR CODE HERE
import sys

N, Q = map(int, input().split())
parts = [(i, 0) for i in range(1, N + 1)]

for _ in range(Q):
    query = input().split()
    if query[0] == '1':
        C = query[1]
        head_x, head_y = parts[0]
        if C == 'R':
            head_x += 1
        elif C == 'L':
            head_x -= 1
        elif C == 'U':
            head_y += 1
        elif C == 'D':
            head_y -= 1
        parts = [(head_x, head_y)] + parts[:-1]
    else:
        p = int(query[1])
        x, y = parts[p - 1]
        print(x, y)