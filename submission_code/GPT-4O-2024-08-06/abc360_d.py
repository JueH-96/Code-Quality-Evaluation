def count_passing_pairs(N, T, S, X):
    # Pair up positions with their directions
    ants = list(zip(X, S))
    # Sort ants based on their positions
    ants.sort()
    
    # Extract sorted positions and directions
    sorted_X = [x for x, _ in ants]
    sorted_S = [s for _, s in ants]
    
    # Two pointers to count valid pairs
    count = 0
    j = 0
    
    # Iterate over each ant
    for i in range(N):
        if sorted_S[i] == '1':  # Only consider ants moving right
            # Move j to find the first ant moving left such that the distance is <= 2T
            while j < N and (sorted_X[j] <= sorted_X[i] or sorted_S[j] != '0' or sorted_X[j] - sorted_X[i] > 2 * T):
                j += 1
            # Count all valid pairs with current i
            k = j
            while k < N and sorted_X[k] - sorted_X[i] <= 2 * T:
                if sorted_S[k] == '0':
                    count += 1
                k += 1
    
    return count

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
T = int(data[1])
S = data[2]
X = list(map(int, data[3:]))

# Calculate and print the result
result = count_passing_pairs(N, T, S, X)
print(result)