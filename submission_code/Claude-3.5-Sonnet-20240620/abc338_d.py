# YOUR CODE HERE
def min_tour_length(N, M, X):
    # Calculate distances between consecutive islands in the tour
    distances = [(X[i] - X[i-1]) % N for i in range(1, M)]
    distances.append((X[0] - X[-1]) % N)
    
    # Calculate the total tour length without closing any bridge
    total_length = sum(min(d, N - d) for d in distances)
    
    # Find the maximum savings by closing a bridge
    max_savings = max(min(d, N - d) - max(d - 1, 1) for d in distances)
    
    # Return the minimum tour length after closing the optimal bridge
    return total_length - max_savings

# Read input
N, M = map(int, input().split())
X = list(map(int, input().split()))

# Calculate and print the result
print(min_tour_length(N, M, X))