import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
intervals = []

index = 1
for i in range(N):
    l = int(data[index])
    r = int(data[index + 1])
    intervals.append((l, r))
    index += 2

intervals.sort(key=lambda x: x[1])  # Sort intervals based on the right endpoint

count = 0
end = -1

for i in range(N):
    l, r = intervals[i]
    if l > end:
        end = r
    else:
        count += 1

print(count)