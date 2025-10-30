def distance_in_steps(vertex1, vertex2):
    """Calculate the minimum number of steps between two vertices in a regular pentagon."""
    vertices = "ABCDE"
    pos1 = vertices.index(vertex1)
    pos2 = vertices.index(vertex2)
    
    # Calculate the distance in both directions
    clockwise = (pos2 - pos1) % 5
    counterclockwise = (pos1 - pos2) % 5
    
    # Return the shorter distance
    return min(clockwise, counterclockwise)

# Read input
s1s2 = input().strip()
t1t2 = input().strip()

s1, s2 = s1s2[0], s1s2[1]
t1, t2 = t1t2[0], t1t2[1]

# Calculate distances in steps
steps_s = distance_in_steps(s1, s2)
steps_t = distance_in_steps(t1, t2)

# Check if distances are the same
if steps_s == steps_t:
    print("Yes")
else:
    print("No")