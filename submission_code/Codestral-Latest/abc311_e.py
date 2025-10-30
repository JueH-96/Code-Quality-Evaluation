# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

H = int(data[0])
W = int(data[1])
N = int(data[2])

holed = set()
for i in range(N):
    a = int(data[3 + 2 * i])
    b = int(data[4 + 2 * i])
    holed.add((a, b))

count = 0

for n in range(1, min(H, W) + 1):
    for i in range(1, H - n + 2):
        for j in range(1, W - n + 2):
            is_holeless = True
            for k in range(n):
                for l in range(n):
                    if (i + k, j + l) in holed:
                        is_holeless = False
                        break
                if not is_holeless:
                    break
            if is_holeless:
                count += 1

print(count)