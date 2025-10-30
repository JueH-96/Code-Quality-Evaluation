def solve():
    n = int(input())
    intervals = []
    for _ in range(n):
        l, r = map(int, input().split())
        intervals.append((l, r))
    
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            l1, r1 = intervals[i]
            l2, r2 = intervals[j]
            if max(l1, l2) <= min(r1, r2):
                count += 1
    print(count)

solve()