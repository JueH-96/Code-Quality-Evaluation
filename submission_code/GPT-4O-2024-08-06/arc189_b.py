def minimize_sum_of_coordinates(N, X):
    # Initial sum of coordinates
    min_sum = sum(X)
    
    # Iterate over each possible group of four consecutive pieces
    for i in range(N - 3):
        # Calculate the midpoint M
        M = (X[i] + X[i + 3]) / 2
        
        # Calculate new positions for X[i+1] and X[i+2]
        new_Xi1 = M + (M - X[i + 1])
        new_Xi2 = M - (X[i + 2] - M)
        
        # Calculate the new sum for this configuration
        current_sum = X[i] + new_Xi1 + new_Xi2 + X[i + 3]
        
        # Update the minimum sum found
        min_sum = min(min_sum, current_sum)
    
    return int(min_sum)

# Reading input
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
X = list(map(int, data[1:]))

# Output the result
print(minimize_sum_of_coordinates(N, X))