def solve():
    n = int(input())
    a = list(map(int, input().split()))

    counts = {}
    for x in a:
        counts[x] = counts.get(x, 0) + 1

    ans = 0
    for count in counts.values():
        ans += count * (count - 1) // 2

    count_1 = counts.get(1, 0)
    count_2 = counts.get(2, 0)
    ans += count_1 * count_2

    print(ans)

t = int(input())
for _ in range(t):
    solve()