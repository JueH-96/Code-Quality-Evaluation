# YOUR CODE HERE
def get_distance_type(p1, p2):
    # Map vertices to positions in pentagon
    vertices = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    
    pos1 = vertices[p1]
    pos2 = vertices[p2]
    
    # Calculate minimum distance around the pentagon
    diff = abs(pos1 - pos2)
    min_diff = min(diff, 5 - diff)
    
    # Adjacent vertices have distance 1, diagonals have distance 2
    return min_diff

# Read input
line1 = input().strip()
line2 = input().strip()

s1, s2 = line1[0], line1[1]
t1, t2 = line2[0], line2[1]

# Get distance types
dist1 = get_distance_type(s1, s2)
dist2 = get_distance_type(t1, t2)

# Compare distances
if dist1 == dist2:
    print("Yes")
else:
    print("No")