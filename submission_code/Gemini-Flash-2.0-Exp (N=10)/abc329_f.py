def solve():
    n, q = map(int, input().split())
    colors = list(map(int, input().split()))
    boxes = [[] for _ in range(n)]
    for i, color in enumerate(colors):
        boxes[i].append(color)

    for _ in range(q):
        a, b = map(int, input().split())
        a -= 1
        b -= 1
        
        boxes[b].extend(boxes[a])
        boxes[a] = []
        
        print(len(set(boxes[b])))

solve()