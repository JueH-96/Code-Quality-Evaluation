# YOUR CODE HERE
import sys

def solve():
    # Read input N
    N = int(sys.stdin.readline())
    # Read initial coordinates X_1, ..., X_N
    X = list(map(int, sys.stdin.readline().split()))

    # Calculate initial differences D_k = X_{k+1} - X_k for k=1 to N-1.
    # We store them in a 0-indexed list D, where D[k] corresponds to the difference D_{k+1}.
    # The list D will have N-1 elements.
    D = []
    for k in range(N - 1):
        D.append(X[k+1] - X[k])

    # Separate the differences based on the parity of their original 1-based index.
    # Differences D_1, D_3, D_5, ... have odd indices. These correspond to elements at indices 0, 2, 4, ... in the 0-based list D.
    # Differences D_2, D_4, D_6, ... have even indices. These correspond to elements at indices 1, 3, 5, ... in the 0-based list D.
    
    D_odd = [] # Stores differences originally at odd indices (1, 3, ...)
    D_even = [] # Stores differences originally at even indices (2, 4, ...)
    for k in range(N - 1):
        # Check the 1-based index (k+1) parity.
        if (k + 1) % 2 != 0: # If the 1-based index k+1 is odd
            D_odd.append(D[k])
        else: # If the 1-based index k+1 is even
            D_even.append(D[k])

    # Sort the differences within each parity group in ascending order.
    # This prepares them for the optimal assignment to minimize the total sum.
    # The minimum sum is achieved when smaller differences are associated with larger weights (N-j) in the sum formula.
    D_odd.sort()
    D_even.sort()

    # Construct the final sequence of differences D_prime.
    # This sequence represents the differences between adjacent pieces in the final configuration that minimizes the sum.
    # We assign the smallest available difference from the correct parity group to each position D'_j (stored at D_prime[j-1]).
    # The position D'_j corresponds to index j-1 in the 0-based list D_prime.
    # The logic derived is to assign the smallest elements of D_odd to D'_1, D'_3, ... and smallest of D_even to D'_2, D'_4, ...
    D_prime = [0] * (N - 1)
    
    idx_odd = 0  # Pointer for the next available smallest difference in D_odd
    idx_even = 0 # Pointer for the next available smallest difference in D_even
    
    # Iterate through positions k from 0 to N-2, corresponding to differences D'_1 to D'_{N-1}
    for k in range(N - 1): 
        # Check the parity of the 1-based index (k+1) for the difference D'_{k+1}
        if (k + 1) % 2 != 0: # If the 1-based index is odd (position D'_1, D'_3, ...)
            # Assign the next smallest available odd difference
            D_prime[k] = D_odd[idx_odd]
            idx_odd += 1
        else: # If the 1-based index is even (position D'_2, D'_4, ...)
            # Assign the next smallest available even difference
            D_prime[k] = D_even[idx_even]
            idx_even += 1

    # Reconstruct the final coordinates X_prime based on the optimal sequence of differences D_prime.
    # The coordinate of the first piece (X_1) remains unchanged throughout the operations.
    X_prime = [0] * N
    X_prime[0] = X[0] 
    
    # Calculate the total sum of final coordinates efficiently.
    # Initialize the sum with the coordinate of the first piece.
    current_sum = X_prime[0] 
    
    # Calculate the remaining coordinates and accumulate the sum iteratively.
    for k in range(N - 1):
        # Calculate the coordinate of the (k+2)-th piece (0-indexed k+1)
        # using the coordinate of the (k+1)-th piece (0-indexed k)
        # and the k-th difference in the optimal sequence D_prime (0-indexed k)
        X_prime[k+1] = X_prime[k] + D_prime[k]
        
        # Add the newly computed coordinate to the total sum
        current_sum += X_prime[k+1]

    # Print the minimum possible sum of coordinates
    print(current_sum)

# Call the solve function to execute the program logic
solve()

# END OF YOUR CODE HERE