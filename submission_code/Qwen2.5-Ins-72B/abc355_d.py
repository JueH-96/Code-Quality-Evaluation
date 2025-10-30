# YOUR CODE HERE
import sys
from bisect import bisect_left, bisect_right

def count_intersecting_pairs(N, intervals):
    intervals.sort()
    count = 0
    for i in range(N):
        li, ri = intervals[i]
        count += bisect_right(intervals, (li, float('inf'))) - (i + 1)
        count += bisect_left(intervals, (ri, 0)) - (i + 1)
    return count

N = int(input())
intervals = []
for _ in range(N):
    l, r = map(int, input().split())
    intervals.append((l, r))

print(count_intersecting_pairs(N, intervals))