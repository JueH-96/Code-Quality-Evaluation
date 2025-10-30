import sys

# Read input
N = int(input())
K = [int(x) for x in input().split()]

# Sort the departments in descending order of size
K.sort(reverse=True)

# Initialize the total number of people in each group
group_a = 0
group_b = 0

# Assign departments to groups
for i in range(N):
    if group_a <= group_b:
        group_a += K[i]
    else:
        group_b += K[i]

# Print the minimum of the maximum number of people in each group
print(min(group_a, group_b))