# YOUR CODE HERE
n = int(input())
intervals = []
for _ in range(n):
    l, r = map(int, input().split())
    intervals.append((l, r))

count = 0
for i in range(n):
    for j in range(i + 1, n):
        # Two intervals [l1, r1] and [l2, r2] intersect if max(l1, l2) <= min(r1, r2)
        if max(intervals[i][0], intervals[j][0]) <= min(intervals[i][1], intervals[j][1]):
            count += 1

print(count)