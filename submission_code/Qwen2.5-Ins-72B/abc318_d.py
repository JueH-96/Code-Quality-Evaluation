import itertools

def max_weight_matching(N, D):
    # Initialize the maximum weight to 0
    max_weight = 0
    
    # Generate all possible subsets of vertices
    for subset in itertools.combinations(range(N), N // 2):
        # Generate all possible pairings of the subset
        for pairing in itertools.permutations(subset):
            # Ensure the pairing is valid (i.e., each vertex is used exactly once)
            if len(set(pairing)) == len(pairing):
                # Calculate the total weight of the current pairing
                current_weight = sum(D[min(i, j)][max(i, j)] for i, j in zip(pairing[::2], pairing[1::2]))
                # Update the maximum weight if the current weight is greater
                max_weight = max(max_weight, current_weight)
    
    return max_weight

# Read input
N = int(input())
D = [[0] * N for _ in range(N)]
for i in range(N - 1):
    row = list(map(int, input().split()))
    for j in range(i + 1, N):
        D[i][j] = D[j][i] = row[j - i - 1]

# Solve the problem
result = max_weight_matching(N, D)

# Print the result
print(result)