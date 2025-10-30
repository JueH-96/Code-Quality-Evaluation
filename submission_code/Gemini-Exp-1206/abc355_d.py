def solve():
    n = int(input())
    intervals = []
    for _ in range(n):
        l, r = map(int, input().split())
        intervals.append((l, r))

    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if max(intervals[i][0], intervals[j][0]) <= min(intervals[i][1], intervals[j][1]):
                count += 1
    print(count)

solve()