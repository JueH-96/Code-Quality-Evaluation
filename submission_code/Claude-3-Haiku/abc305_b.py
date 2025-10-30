# Read the input from stdin
p, q = input().split()

# Define the distances between adjacent points
distances = [3, 1, 4, 1, 5, 9]

# Find the index of the points p and q
p_index = ord(p) - ord('A')
q_index = ord(q) - ord('A')

# Calculate the distance between the points
distance = 0
if p_index < q_index:
    for i in range(p_index, q_index):
        distance += distances[i]
else:
    for i in range(q_index, p_index):
        distance += distances[i]

# Print the result
print(distance)