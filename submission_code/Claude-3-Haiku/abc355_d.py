# Read the input
N = int(input())
intervals = []
for _ in range(N):
    l, r = map(int, input().split())
    intervals.append((l, r))

# Sort the intervals based on the start time
intervals.sort(key=lambda x: x[0])

# Count the number of intersecting pairs
count = 0
for i in range(N):
    for j in range(i+1, N):
        if intervals[i][1] > intervals[j][0]:
            count += 1

# Print the answer
print(count)