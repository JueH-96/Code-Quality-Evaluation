# YOUR CODE HERE
import sys

# Read input from stdin
p, q = sys.stdin.read().split()

# Define the distances between adjacent points
distances = {
    ('A', 'B'): 3,
    ('B', 'C'): 1,
    ('C', 'D'): 4,
    ('D', 'E'): 1,
    ('E', 'F'): 5,
    ('F', 'G'): 9
}

# Calculate the distance between p and q
if p < q:
    distance = sum(distances[(p, p+chr(ord(i)+1))] for i in range(ord(p), ord(q)))
else:
    distance = sum(distances[(q, q+chr(ord(i)+1))] for i in range(ord(q), ord(p)))

# Print the result
print(distance)