import bisect

n = int(input())
intervals = []
for _ in range(n):
    l, r = map(int, input().split())
    intervals.append((l, r))

intervals.sort(key=lambda x: x[1])
rs = []
total = 0

for l, r in intervals:
    pos = bisect.bisect_left(rs, l)
    total += len(rs) - pos
    rs.append(r)

print(total)