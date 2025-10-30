def solve():
    n, m = map(int, input().split())
    boxes = []
    for _ in range(n):
        boxes.append(tuple(map(int, input().split())))
    
    boxes.sort(key=lambda x: x[1])
    
    ans = 0
    
    for i in range(1 << n):
        cost = 0
        capacity = 0
        
        for j in range(n):
            if (i >> j) & 1:
                cost += boxes[j][1]
                capacity += boxes[j][0]
        
        ans = max(ans, capacity - cost)
    
    print(ans)

t = int(input())
for _ in range(t):
    solve()