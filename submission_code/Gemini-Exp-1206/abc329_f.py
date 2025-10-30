def solve():
    n, q = map(int, input().split())
    colors = list(map(int, input().split()))
    boxes = [set() for _ in range(n)]
    for i in range(n):
        boxes[i].add(colors[i])

    for _ in range(q):
        a, b = map(int, input().split())
        a -= 1
        b -= 1
        boxes[b].update(boxes[a])
        boxes[a].clear()
        print(len(boxes[b]))

solve()