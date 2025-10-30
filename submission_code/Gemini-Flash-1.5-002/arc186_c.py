def solve():
    N, M = map(int, input().split())
    boxes = []
    for _ in range(N):
        boxes.append(list(map(int, input().split())))

    boxes.sort(key=lambda x: x[1] / x[0])

    ans = 0
    cost = 0
    
    
    for i in range(N):
        capacity, price = boxes[i]
        ans += capacity
        cost += price

    print(min(ans, cost))


T = int(input())
for _ in range(T):
    solve()