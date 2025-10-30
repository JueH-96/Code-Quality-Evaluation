import math

# Define the coordinates of the vertices of the regular pentagon
vertices = {
    'A': (0, 1),
    'B': (math.cos(math.pi/2.5), math.sin(math.pi/2.5)),
    'C': (math.cos(math.pi/2.5*2), math.sin(math.pi/2.5*2)),
    'D': (math.cos(math.pi/2.5*3), math.sin(math.pi/2.5*3)),
    'E': (math.cos(math.pi/2.5*4), math.sin(math.pi/2.5*4)),
}

def distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def main():
    # Read the input from stdin
    s1, s2 = input().strip()
    t1, t2 = input().strip()

    # Calculate the distances
    dist_s = distance(vertices[s1], vertices[s2])
    dist_t = distance(vertices[t1], vertices[t2])

    # Print the result
    if math.isclose(dist_s, dist_t):
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()