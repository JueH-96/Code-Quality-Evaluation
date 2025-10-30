# Read input
N = int(input())
P = list(map(int, input().split()))

# Person 1's programming ability
P_1 = P[0]

# Find the maximum programming ability among others
max_others = 0
for i in range(1, N):
    max_others = max(max_others, P[i])

# Calculate how many more points person 1 needs
# If P_1 is already greater than max_others, person 1 needs 0 more points
# Otherwise, person 1 needs (max_others - P_1 + 1) points to exceed the maximum
points_needed = max(0, max_others - P_1 + 1)

# Print the answer
print(points_needed)