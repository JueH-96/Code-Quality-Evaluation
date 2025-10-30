# YOUR CODE HERE
import sys

# Function to read input faster, standard practice in competitive programming
# `strip()` removes potential leading/trailing whitespace including newline
def I(): return sys.stdin.readline().strip()

def solve():
    # Read integer N, the length of the string S
    N = int(I())
    # Read string S consisting of digits '1' through '9'
    S = I()
    
    # Initialize the DP value accumulator. `current_dp_val` represents DP[k] at the end of iteration k.
    # DP[k] stores the sum of numerical values of all substrings ending at index k.
    # Based on the recurrence DP[k] = 10 * DP[k-1] + (k+1) * int(S[k]).
    # Base case DP[-1] = 0 is implicitly handled by initializing current_dp_val = 0.
    current_dp_val = 0 
    
    # Initialize the total sum accumulator. This variable will store the final answer,
    # which is the sum of DP[k] for all k from 0 to N-1.
    total_sum = 0
    
    # Iterate through the string S with index k ranging from 0 to N-1
    for k in range(N):
        # Convert the character S[k] (which is a digit) to its integer value
        digit = int(S[k])
        
        # Apply the DP recurrence relation to compute DP[k]:
        # DP[k] = 10 * DP[k-1] + (k+1) * digit
        # The variable `current_dp_val` holds the value of DP[k-1] before this line.
        # After this line, `current_dp_val` is updated to hold the value of DP[k].
        # Python's built-in `int` type handles arbitrary precision integers, which is necessary
        # as the values can become very large.
        current_dp_val = current_dp_val * 10 + (k + 1) * digit
        
        # Accumulate the computed DP[k] (stored in `current_dp_val`) into the total sum.
        # The total sum after iteration k is equal to sum_{p=0}^{k} DP[p].
        total_sum += current_dp_val
        
    # After the loop finishes, `total_sum` holds the sum over all k from 0 to N-1, 
    # which is the required answer: sum_{i=0}^{N-1} sum_{j=i}^{N-1} f(i, j).
    # Print the final result to standard output.
    print(total_sum)

# Call the solve function to execute the program logic
solve()
# YOUR CODE HERE