def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    positions = {}
    for i in range(2 * n):
        if a[i] not in positions:
            positions[a[i]] = []
        positions[a[i]].append(i)
    
    count = 0
    for color in range(1, n + 1):
        if positions[color][1] - positions[color][0] == 2:
            count += 1
    
    print(count)

solve()