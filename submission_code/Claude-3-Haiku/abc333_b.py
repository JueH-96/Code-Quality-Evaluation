# Read the input from stdin
s1, s2 = input().split()
t1, t2 = input().split()

# Define the coordinates of the pentagon vertices
vertices = {'A': (0, 1), 'B': (0.9239, 0.3827), 'C': (0.9239, -0.3827), 'D': (0, -1), 'E': (-0.9239, -0.3827)}

# Calculate the distance between two points
def distance(p1, p2):
    x1, y1 = vertices[p1]
    x2, y2 = vertices[p2]
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

# Check if the distances are equal
if distance(s1, s2) == distance(t1, t2):
    print("Yes")
else:
    print("No")