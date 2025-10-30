def solve():
    n, m = map(int, input().split())
    x = list(map(int, input().split()))
    a = list(map(int, input().split()))

    stones = {}
    for i in range(m):
        stones[x[i]] = a[i]

    total_stones = sum(a)
    if total_stones != n:
        print(-1)
        return

    operations = 0
    current_stones = 0
    for i in range(1, n + 1):
        current_stones += stones.get(i, 0)
        if current_stones > 1:
            operations += current_stones - 1
            current_stones = 1
        elif current_stones < 1:
            print(-1)
            return
    
    print(operations)

solve()