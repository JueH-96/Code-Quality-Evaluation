def solve():
    n = int(input())
    giants = []
    for _ in range(n):
        giants.append(list(map(int, input().split())))

    giants.sort(key=lambda x: x[0])
    
    total_a = 0
    max_head_height = 0
    for i in range(n):
        total_a += giants[i][0]
        max_head_height = max(max_head_height, total_a - giants[i][0] + giants[i][1])

    print(max_head_height)

solve()