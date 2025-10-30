# YOUR CODE HERE

def calculate_distance(p, q):
    distances = {
        'A': {'B': 3},
        'B': {'C': 1},
        'C': {'D': 4},
        'D': {'E': 1},
        'E': {'F': 5},
        'F': {'G': 9},
    }
    return distances[p][q]

p, q = input().split()
print(calculate_distance(p, q))