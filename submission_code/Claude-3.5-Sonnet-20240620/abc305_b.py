# YOUR CODE HERE
def calculate_distance(p, q):
    points = 'ABCDEFG'
    distances = [3, 1, 4, 1, 5, 9]
    
    start = min(points.index(p), points.index(q))
    end = max(points.index(p), points.index(q))
    
    return sum(distances[start:end])

p, q = input().split()
print(calculate_distance(p, q))