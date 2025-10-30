def solve():
    n, m = map(int, input().split())
    x = list(map(int, input().split()))
    a = list(map(int, input().split()))

    total_stones = sum(a)
    if total_stones != n:
        print(-1)
        return

    initial_stones = [0] * (n + 1)
    for i in range(m):
        initial_stones[x[i]] = a[i]

    moves = 0
    cumulative_stones = 0
    for i in range(1, n + 1):
        cumulative_stones += initial_stones[i]
        if cumulative_stones < i:
            print(-1)
            return
        if i < n:
            moves += cumulative_stones - i

    print(moves)

solve()