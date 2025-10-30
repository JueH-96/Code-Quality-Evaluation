# YOUR CODE HERE
import itertools

def max_weight(N, weights):
    max_sum = 0
    vertices = list(range(N))
    
    # Generate all possible pairs of vertices
    pairs = list(itertools.combinations(vertices, 2))
    
    # Generate all possible ways to choose pairs such that no vertex is repeated
    for pair_set in itertools.combinations(pairs, N // 2):
        used_vertices = set()
        valid = True
        current_sum = 0
        
        for u, v in pair_set:
            if u in used_vertices or v in used_vertices:
                valid = False
                break
            used_vertices.add(u)
            used_vertices.add(v)
            current_sum += weights[min(u, v)][max(u, v) - min(u, v) - 1]
        
        if valid:
            max_sum = max(max_sum, current_sum)
    
    return max_sum

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
weights = []
index = 1
for i in range(1, N):
    weights.append([int(data[index + j]) for j in range(N - i)])
    index += (N - i)

# Calculate and print the result
print(max_weight(N, weights))