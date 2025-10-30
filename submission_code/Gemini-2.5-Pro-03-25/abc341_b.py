# YOUR CODE HERE
import sys

def solve():
    # Read the number of countries
    N = int(sys.stdin.readline())
    
    # Read the initial amounts of currency for each country
    # A is a list of N integers, 0-indexed. A[i] corresponds to country i+1.
    A = list(map(int, sys.stdin.readline().split()))
    
    # Read the N-1 conversion rate pairs (S_i, T_i)
    # S_vals[i] and T_vals[i] correspond to the operation between
    # country i+1 and country i+2.
    # This operation uses currency of country i+1 (A[i])
    # costs S_vals[i] per operation, and gains T_vals[i] of currency of country i+2 (A[i+1]).
    S_vals = []
    T_vals = []
    for _ in range(N - 1):
        s, t = map(int, sys.stdin.readline().split())
        S_vals.append(s)
        T_vals.append(t)

    # Iterate through each possible conversion step.
    # The loop index `i` goes from 0 to N-2.
    # The i-th iteration handles the conversion from country i+1 to country i+2.
    for i in range(N - 1):
        # Check if Takahashi has enough currency of country i+1 (A[i]) to perform the operation at least once.
        # The cost per operation is S_vals[i].
        # This check `if A[i] >= S_vals[i]:` is technically not needed because
        # integer division `//` handles the case where A[i] < S_vals[i] by resulting in 0.
        # But let's proceed without the explicit check for simplicity.

        # Calculate the maximum number of times the operation can be performed.
        # `A[i] // S_vals[i]` computes floor(A[i] / S_vals[i]), which is the maximum number of times
        # Takahashi can pay S_vals[i] units from his A[i] units.
        num_operations = A[i] // S_vals[i]
        
        # If num_operations > 0, it means the operation can be performed.
        # Calculate the total currency gained for country i+2 (A[i+1]).
        # Each operation yields T_vals[i] units.
        # The total gained is num_operations * T_vals[i].
        # Add this gained amount to the currency balance of country i+2.
        # If num_operations is 0, this adds 0, which correctly reflects that no currency is gained.
        A[i+1] += num_operations * T_vals[i]
        
        # Note: We don't need to update A[i] (subtract the spent currency)
        # because the amount A[i] is not used in subsequent iterations of THIS loop
        # for any calculation affecting A[N-1]. The loop moves forward i -> i+1 -> ...
        # and A[i] is only used in the i-th iteration to calculate gain for A[i+1].

    # After the loop finishes, A[N-1] holds the final amount of currency for country N.
    # Print the result.
    print(A[N-1])

# Execute the main solution function
solve()

# END OF YOUR CODE HERE