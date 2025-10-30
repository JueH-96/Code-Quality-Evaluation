# Read the input
N = int(input())
P = list(map(int, input().split()))

# Find the minimum points needed for person 1 to become the strongest
min_points = 0
for i in range(1, N):
    if P[0] <= P[i]:
        min_points = max(min_points, P[i] - P[0] + 1)

# Print the answer
print(min_points)