import sys

def solve():
    # Read the integer N, the length of the sequence/string
    N = int(sys.stdin.readline())
    # Read the string S consisting of '0' and '1'
    S = sys.stdin.readline().strip()

    # Initialize the total sum we want to compute
    total_sum = 0
    
    # Initialize T_prev, which represents T_{j-1} in the recurrence relation.
    # We use 0-based indexing for the string S and our DP state T.
    # The recurrence for T_j depends on T_{j-1}.
    # For the base case j=0, we need T_{-1}. We derived the recurrence such that T_{-1}=0 works.
    T_prev = 0 

    # Iterate through the indices j from 0 to N-1.
    # In each iteration j, we compute T_j = sum_{i=0}^j f(i, j)
    # and add it to the total_sum.
    for j in range(N):
        # Get the character S[j]. This corresponds to A_j in the problem statement (adjusted for 0-based index).
        # If S[j] is '0', this means A_j = 0.
        if S[j] == '0':
            # According to the derived recurrence relation for 0-based indexing:
            # If A_j = 0, then T_j = j.
            T_curr = j
        # If S[j] is '1', this means A_j = 1.
        else: # S[j] == '1'
            # According to the derived recurrence relation for 0-based indexing:
            # If A_j = 1, then T_j = 1 + j - T_{j-1}.
            # T_prev holds the value of T_{j-1} from the previous iteration.
            T_curr = 1 + j - T_prev
        
        # Add the computed value T_j (stored in T_curr) to the total sum.
        # The final answer is the sum of all T_j for j from 0 to N-1.
        total_sum += T_curr
        
        # Update T_prev to T_curr so it holds the correct value T_{j}
        # for the next iteration (which will be computing T_{j+1}).
        T_prev = T_curr

    # After the loop finishes, total_sum holds the final answer.
    # Print the answer to standard output.
    print(total_sum)

# Call the solve function to execute the main logic of the program.
solve()