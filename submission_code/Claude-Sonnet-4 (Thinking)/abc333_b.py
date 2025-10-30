# Read input
s1s2 = input().strip()
t1t2 = input().strip()

s1, s2 = s1s2[0], s1s2[1]
t1, t2 = t1t2[0], t1t2[1]

# Map vertices to indices
vertex_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

# Calculate circular distance
def circular_distance(v1, v2):
    i = vertex_to_index[v1]
    j = vertex_to_index[v2]
    return min(abs(i - j), 5 - abs(i - j))

# Calculate distances
dist1 = circular_distance(s1, s2)
dist2 = circular_distance(t1, t2)

# Compare and output
if dist1 == dist2:
    print("Yes")
else:
    print("No")