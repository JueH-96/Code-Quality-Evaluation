# YOUR CODE HERE
def find_distance(p, q):
    distances = {
        ('A', 'B'): 3,
        ('B', 'C'): 1,
        ('C', 'D'): 4,
        ('D', 'E'): 1,
        ('E', 'F'): 5,
        ('F', 'G'): 9
    }
    
    points = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    
    # Ensure p comes before q in the list
    if points.index(p) > points.index(q):
        p, q = q, p
    
    total_distance = 0
    for i in range(points.index(p), points.index(q)):
        total_distance += distances[(points[i], points[i+1])]
    
    return total_distance

# Read input
import sys
input = sys.stdin.read().strip()
p, q = input.split()

# Calculate and print the distance
print(find_distance(p, q))