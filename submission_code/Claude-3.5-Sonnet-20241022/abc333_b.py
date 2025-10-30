s1s2 = input().strip()
t1t2 = input().strip()

# Create a dictionary to map vertices to their distances
distances = {
    'AB': 1, 'BC': 1, 'CD': 1, 'DE': 1, 'EA': 1,  # Adjacent vertices
    'AC': 2, 'BD': 2, 'CE': 2, 'DA': 2, 'EB': 2,  # Vertices with one vertex between
    'AD': 2, 'BE': 2, 'CA': 2, 'DB': 2, 'EC': 2   # Same as above, reverse order
}

# Function to get distance between two vertices
def get_distance(v1, v2):
    if v1 + v2 in distances:
        return distances[v1 + v2]
    return distances[v2 + v1]

# Get distances for both pairs
dist1 = get_distance(s1s2[0], s1s2[1])
dist2 = get_distance(t1t2[0], t1t2[1])

# Compare and output result
print("Yes" if dist1 == dist2 else "No")