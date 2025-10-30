# Read input values
N_1, N_2, M = map(int, input().split())

# Since all vertices in N_1 are connected and all vertices in N_2 are connected,
# the maximum distance is achieved by connecting the farthest vertices in each group.
# This is equivalent to connecting any vertex in N_1 to any vertex in N_2.
# The maximum distance will be the sum of the sizes of the two groups minus 1
# (since we add one edge between the two groups).

# Calculate the maximum possible distance
max_distance = N_1 + N_2 - 1

# Output the result
print(max_distance)