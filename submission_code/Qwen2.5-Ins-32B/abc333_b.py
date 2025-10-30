def is_same_length(s1, s2, t1, t2):
    # Define the order of the vertices in the pentagon
    vertices = ['A', 'B', 'C', 'D', 'E']
    
    # Function to calculate the distance between two points in the pentagon
    def distance(p1, p2):
        i1 = vertices.index(p1)
        i2 = vertices.index(p2)
        return min(abs(i1 - i2), 5 - abs(i1 - i2))
    
    # Calculate distances
    dist_s = distance(s1, s2)
    dist_t = distance(t1, t2)
    
    # Compare distances
    return dist_s == dist_t

# Read input
s1, s2 = input().strip()
t1, t2 = input().strip()

# Check if the lengths are the same and print the result
if is_same_length(s1, s2, t1, t2):
    print("Yes")
else:
    print("No")