import sys

def solve():
    # Read N and D from the first line of input
    N, D = map(int, sys.stdin.readline().split())

    # Store the initial thickness (T) and length (L) for each snake
    snakes = []
    for _ in range(N):
        T, L = map(int, sys.stdin.readline().split())
        snakes.append((T, L)) # Store as a tuple (thickness, initial_length)

    # Iterate for each value of k from 1 to D
    for k in range(1, D + 1):
        max_weight_for_k = 0 # Initialize the maximum weight for the current k

        # Iterate through each snake to calculate its weight for the current k
        for T_i, L_i in snakes:
            # Calculate the new length for the current snake
            current_length = L_i + k
            # Calculate the weight of the current snake
            current_weight = T_i * current_length
            
            # Update max_weight_for_k if the current snake's weight is greater
            if current_weight > max_weight_for_k:
                max_weight_for_k = current_weight
        
        # Print the maximum weight found for the current k
        print(max_weight_for_k)

# This ensures that the solve() function is called only when the script is executed directly
if __name__ == '__main__':
    solve()