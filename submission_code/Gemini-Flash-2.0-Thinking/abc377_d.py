def solve():
    n, m = map(int, input().split())
    intervals = []
    for _ in range(n):
        l_i, r_i = map(int, input().split())
        intervals.append((l_i, r_i))

    count = 0
    for l in range(1, m + 1):
        for r in range(l, m + 1):
            satisfies = True
            for li, ri in intervals:
                if l <= li and r >= ri:
                    satisfies = False
                    break
            if satisfies:
                count += 1
    print(count)

solve()