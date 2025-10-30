import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
D = int(data[1])

points = []
for i in range(N):
    x = int(data[2 + 2 * i])
    y = int(data[3 + 2 * i])
    points.append((x, y))

def count_points(D):
    count = 0
    for x in range(-10**6, 10**6 + 1):
        for y in range(-10**6, 10**6 + 1):
            if sum(abs(x - px) + abs(y - py) for px, py in points) <= D:
                count += 1
    return count

print(count_points(D))