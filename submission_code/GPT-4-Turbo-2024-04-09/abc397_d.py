def find_xy(N):
    x = 1
    while True:
        x_cubed = x ** 3
        if x_cubed > N + 1:
            break
        y_cubed = x_cubed - N
        y = int(round(y_cubed ** (1/3)))
        if y > 0 and x ** 3 - y ** 3 == N:
            return x, y
        x += 1
    return -1

import sys
input = sys.stdin.read
N = int(input().strip())
result = find_xy(N)
if result == -1:
    print(-1)
else:
    print(result[0], result[1])