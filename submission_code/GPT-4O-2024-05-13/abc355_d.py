# YOUR CODE HERE
import sys
input = sys.stdin.read

def count_intersecting_intervals(N, intervals):
    intervals.sort()
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            if intervals[i][1] > intervals[j][0]:
                count += 1
            else:
                break
    return count

data = input().split()
N = int(data[0])
intervals = []
for i in range(N):
    l = int(data[2 * i + 1])
    r = int(data[2 * i + 2])
    intervals.append((l, r))

print(count_intersecting_intervals(N, intervals))