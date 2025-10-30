def solve():
    n, m = map(int, input().split())
    x = list(map(int, input().split()))
    a = list(map(int, input().split()))

    total_stones = sum(a)
    if total_stones != n:
        print("-1")
        return

    stones = {}
    for i in range(m):
        stones[x[i]] = a[i]

    cells = [0] * (n + 1)
    for pos in stones:
        cells[pos] = stones[pos]

    ans = 0
    needed = 1
    for i in range(1, n + 1):
        if cells[i] >= needed:
            ans += (cells[i] - needed) * (i - 1)
            needed = cells[i]
        else:
            ans += (needed - cells[i]) * (i - 1)
            needed = cells[i]
        needed = 1
    
    
    ans = 0
    current_stones = 0
    for i in range(1, n + 1):
        if i in stones:
            current_stones += stones[i]
        
        ans += abs(current_stones - i)
        
    print(ans)

solve()