# Read the number of people
N = int(input())

# Read the programming abilities
P = list(map(int, input().split()))

# Calculate the points needed for person 1 to be the strongest
additional_points = max(0, max(P[1:]) - P[0] + 1)

# Output the result
print(additional_points)