# YOUR CODE HERE
def find_min_good_index_sequence(N, K, P):
    # Create a dictionary to store the positions of each element in P
    pos = {P[i]: i for i in range(N)}
    
    min_diff = float('inf')
    
    # Iterate through all possible starting points of the consecutive K integers
    for a in range(1, N - K + 2):
        indices = [pos[a + j] for j in range(K)]
        indices.sort()
        min_diff = min(min_diff, indices[-1] - indices[0])
    
    return min_diff

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
K = int(data[1])
P = list(map(int, data[2:]))

# Find and print the result
print(find_min_good_index_sequence(N, K, P))