# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
intervals = []

for i in range(N):
    l = int(data[2*i+1])
    r = int(data[2*i+2])
    intervals.append((l, r))

count = 0
for i in range(N):
    for j in range(i+1, N):
        if intervals[i][0] < intervals[j][1] and intervals[j][0] < intervals[i][1]:
            count += 1

print(count)