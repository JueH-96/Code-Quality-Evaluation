# Read the number of intervals
N = int(input())

# Read the intervals
intervals = [tuple(map(int, input().split())) for _ in range(N)]

# Initialize the count of intersecting pairs
intersecting_pairs = 0

# Check each pair of intervals for intersection
for i in range(N):
    for j in range(i + 1, N):
        if intervals[i][1] >= intervals[j][0] and intervals[j][1] >= intervals[i][0]:
            intersecting_pairs += 1

# Print the result
print(intersecting_pairs)