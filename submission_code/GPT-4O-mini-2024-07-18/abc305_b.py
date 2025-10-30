# YOUR CODE HERE
def calculate_distance(p, q):
    points = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    distances = [0, 3, 1, 4, 1, 5, 9]  # distances between adjacent points
    
    # Create a cumulative distance array
    cumulative_distance = [0] * len(points)
    for i in range(1, len(points)):
        cumulative_distance[i] = cumulative_distance[i - 1] + distances[i - 1]
    
    # Get the indices of p and q
    index_p = points.index(p)
    index_q = points.index(q)
    
    # Calculate the distance
    distance = abs(cumulative_distance[index_p] - cumulative_distance[index_q])
    return distance

# Read input
import sys
input_data = sys.stdin.read().strip()
p, q = input_data.split()

# Calculate and print the distance
result = calculate_distance(p, q)
print(result)