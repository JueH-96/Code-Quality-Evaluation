# YOUR CODE HERE
import sys

# Function to read input and solve the problem
def solve():
    # Read N, D, P from the first line
    # N: number of days
    # D: number of passes per batch
    # P: cost per batch of passes
    N, D, P = map(int, sys.stdin.readline().split())
    
    # Read the N fares F_i from the second line
    F = list(map(int, sys.stdin.readline().split()))

    # Sort the fares in descending order
    # This is because if we decide to use k passes, it's always optimal
    # to use them on the k days with the highest fares to maximize savings.
    F.sort(reverse=True)

    # Calculate prefix sums of the sorted fares
    # prefix_sum_F[k] will store the sum of the top k fares (F[0]...F[k-1])
    # This helps compute the sum of remaining fares efficiently later.
    prefix_sum_F = [0] * (N + 1)
    for i in range(N):
        prefix_sum_F[i+1] = prefix_sum_F[i] + F[i]

    # total_sum_F is the sum of all fares. This is needed to calculate
    # the sum of remaining fares (those not covered by passes).
    total_sum_F = prefix_sum_F[N]
    
    # Initialize minimum total cost with the cost of paying all fares.
    # This corresponds to the case where k=0 passes are used.
    min_total_cost = total_sum_F 

    # Iterate through all possible number of passes k to use, from 1 to N.
    # We check each possibility to find the minimum total cost.
    # The case k=0 is already covered by the initialization of min_total_cost.
    for k in range(1, N + 1):
        # Calculate the number of batches needed to get at least k passes.
        # Since passes are sold in batches of D, we need ceil(k/D) batches.
        # In integer arithmetic, ceil(k/D) can be calculated as (k + D - 1) // D.
        num_batches = (k + D - 1) // D
        
        # Calculate the total cost of purchasing these batches.
        pass_cost = num_batches * P
        
        # Calculate the cost of regular fares for the remaining N-k days.
        # Since we use passes for the k days with the highest fares (sum = prefix_sum_F[k]),
        # the remaining N-k days are those with the lowest fares.
        # The sum of fares for these remaining days is total_sum_F - prefix_sum_F[k].
        fare_cost = total_sum_F - prefix_sum_F[k]
        
        # Calculate the total cost for this choice of k (using k passes)
        current_total_cost = pass_cost + fare_cost
        
        # Update the minimum total cost found so far if the current cost is lower.
        min_total_cost = min(min_total_cost, current_total_cost)

    # Print the final minimum total cost found after checking all possibilities for k.
    print(min_total_cost)

# Call the solve function to run the program
solve()
# END OF YOUR CODE HERE