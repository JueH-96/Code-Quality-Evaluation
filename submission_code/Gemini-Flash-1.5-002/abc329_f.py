# YOUR CODE HERE
n, q = map(int, input().split())
c = list(map(int, input().split()))
boxes = [[] for _ in range(n)]
for i in range(n):
    boxes[i].append(c[i])

for _ in range(q):
    a, b = map(int, input().split())
    a -= 1
    b -= 1
    if len(boxes[a]) > 0:
        boxes[b].extend(boxes[a])
        boxes[a] = []
    print(len(set(boxes[b])))