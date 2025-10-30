# YOUR CODE HERE
def calculate_distance(p, q):
    # Define the distances between adjacent points
    distances = {
        ('A', 'B'): 3,
        ('B', 'C'): 1,
        ('C', 'D'): 4,
        ('D', 'E'): 1,
        ('E', 'F'): 5,
        ('F', 'G'): 9
    }
    
    # Create a list of points in order
    points = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    
    # Find the indices of p and q
    index_p = points.index(p)
    index_q = points.index(q)
    
    # Ensure index_p is less than index_q for easier calculation
    if index_p > index_q:
        index_p, index_q = index_q, index_p
    
    # Calculate the total distance
    total_distance = 0
    for i in range(index_p, index_q):
        total_distance += distances[(points[i], points[i + 1])]
    
    return total_distance

# Read input
import sys
input = sys.stdin.read().strip()
p, q = input.split()

# Calculate and print the distance
print(calculate_distance(p, q))