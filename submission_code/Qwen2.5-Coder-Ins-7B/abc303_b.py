# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

photos = []
index = 2
for _ in range(M):
    row = list(map(int, data[index:index + N]))
    photos.append(row)
    index += N

bad_pairs = 0
for i in range(1, N + 1):
    for j in range(i + 1, N + 1):
        seen = False
        for k in range(M):
            if i in photos[k] and j in photos[k]:
                if abs(photos[k].index(i) - photos[k].index(j)) == 1:
                    seen = True
                    break
        if not seen:
            bad_pairs += 1

print(bad_pairs)