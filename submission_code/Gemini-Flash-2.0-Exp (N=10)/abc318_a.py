def solve():
    n, m, p = map(int, input().split())
    count = 0
    current_day = m
    while current_day <= n:
        count += 1
        current_day += p
    print(count)

solve()