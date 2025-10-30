# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
N = int(data[2])

holes = set()
for i in range(N):
    a = int(data[3 + 2 * i]) - 1
    b = int(data[4 + 2 * i]) - 1
    holes.add((a, b))

def is_holeless(i, j, n):
    for k in range(n):
        for l in range(n):
            if (i + k, j + l) in holes:
                return False
    return True

count = 0
for i in range(H):
    for j in range(W):
        for n in range(1, min(H - i, W - j) + 1):
            if is_holeless(i, j, n):
                count += 1

print(count)