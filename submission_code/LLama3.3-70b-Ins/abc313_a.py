# Read the number of people
N = int(input())

# Read the programming abilities
P = list(map(int, input().split()))

# Find the maximum programming ability excluding person 1
max_ability = max(P[1:])

# Calculate the minimum points needed for person 1 to become the strongest
min_points_needed = max(0, max_ability - P[0] + 1)

# Print the answer
print(min_points_needed)