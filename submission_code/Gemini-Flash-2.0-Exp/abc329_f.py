def solve():
    n, q = map(int, input().split())
    colors = list(map(int, input().split()))
    
    boxes = [[] for _ in range(n + 1)]
    for i in range(n):
        boxes[i+1].append(colors[i])
    
    for _ in range(q):
        a, b = map(int, input().split())
        
        for color in boxes[a]:
            boxes[b].append(color)
        boxes[a] = []
        
        unique_colors = set(boxes[b])
        print(len(unique_colors))

solve()