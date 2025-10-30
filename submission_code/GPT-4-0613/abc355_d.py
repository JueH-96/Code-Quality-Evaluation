import sys
from operator import itemgetter

N = int(sys.stdin.readline())
intervals = []
for _ in range(N):
    l, r = map(int, sys.stdin.readline().split())
    intervals.append((l, r))

intervals.sort(key=itemgetter(1))

count = 0
for i in range(N):
    for j in range(i+1, N):
        if intervals[i][1] > intervals[j][0]:
            count += 1
        else:
            break
print(count)