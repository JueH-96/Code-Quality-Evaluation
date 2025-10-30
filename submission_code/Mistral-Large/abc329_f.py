import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])

colors = [0] + [int(data[i]) for i in range(2, N + 2)]
boxes = [set() for _ in range(N + 1)]

for i in range(1, N + 1):
    boxes[i].add(colors[i])

index = N + 2
results = []

for _ in range(Q):
    a = int(data[index])
    b = int(data[index + 1])
    index += 2

    boxes[b].update(boxes[a])
    boxes[a].clear()

    results.append(len(boxes[b]))

sys.stdout.write("
".join(map(str, results)) + "
")