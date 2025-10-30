N = int(input())
intervals = []
for _ in range(N):
    l, r = map(int, input().split())
    intervals.append((l, r))

count = 0
for i in range(N):
    for j in range(i+1, N):
        # Two intervals [a,b] and [c,d] intersect if max(a,c) <= min(b,d)
        if max(intervals[i][0], intervals[j][0]) <= min(intervals[i][1], intervals[j][1]):
            count += 1

print(count)