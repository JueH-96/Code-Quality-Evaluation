def solve():
    n, m = map(int, input().split())
    intervals = []
    for _ in range(n):
        intervals.append(list(map(int, input().split())))

    count = 0
    for l in range(1, m + 1):
        for r in range(l, m + 1):
            valid = True
            for li, ri in intervals:
                if l <= li and ri <= r:
                    valid = False
                    break
            if valid:
                count += 1
    print(count)

solve()