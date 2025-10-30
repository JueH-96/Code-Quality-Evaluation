# YOUR CODE HERE
import sys

# Function to read input and solve the problem
def solve():
    # Read the number of elements N from standard input
    N = int(sys.stdin.readline())
    
    # Read the sequence A as a list of integers from standard input
    A = list(map(int, sys.stdin.readline().split()))

    # Calculate the sum of elements in the sequence A.
    # Python's arbitrary precision integers handle potentially large sums.
    S = sum(A)
    
    # Determine the target values for the elements in the final sequence.
    # After performing operations, the difference between the maximum and minimum values must be at most 1.
    # This means all elements must be equal to some value k, or some are k and others are k+1.
    # The sum of elements remains constant throughout the operations.
    # Let the final sequence have `N - rem` elements equal to `k` and `rem` elements equal to `k+1`.
    # The sum S must satisfy S = (N - rem) * k + rem * (k+1) = N*k + rem.
    # Thus, k = floor(S / N) and rem = S % N.
    
    # Calculate the base value k (floor of the average)
    k = S // N
    
    # Calculate the remainder, which is the number of elements that should be k+1 in the target state.
    rem = S % N  # This is equivalent to S - N * k for non-negative S, N
    
    # The number of elements that should end up as k in the target state is N - rem.
    
    # Sort the initial array A in ascending order. This allows us to apply a greedy strategy
    # for assigning target values to minimize operations.
    # The optimal strategy assigns the target value k to the N-rem smallest elements 
    # of A and the target value k+1 to the rem largest elements of A.
    A.sort()
    
    # The minimum number of operations required is the total amount of value that needs to be moved.
    # This can be calculated as the sum of all necessary decreases (total excess)
    # OR the sum of all necessary increases (total deficit). These two sums are equal.
    # We will calculate the total excess: the sum of (A[i] - target_value) for all A[i] > target_value.
    
    total_ops = 0
    
    # Calculate the total decrease needed for the first N - rem elements.
    # Their target value is k. These elements correspond to indices 0 to N-rem-1 in the sorted list A.
    # An element A[i] needs decrease if its value A[i] is greater than its target k.
    # The required decrease amount for such an element is A[i] - k.
    # We sum these positive differences.
    for i in range(N - rem):
        # The condition `if A[i] > k` ensures we only sum positive differences (excesses).
        # This is equivalent to summing max(0, A[i] - k).
        if A[i] > k:
            total_ops += A[i] - k
            
    # Calculate the total decrease needed for the last rem elements.
    # Their target value is k+1. These elements correspond to indices N-rem to N-1 in the sorted list A.
    # An element A[i] needs decrease if its value A[i] is greater than its target k + 1.
    # The required decrease amount for such an element is A[i] - (k+1).
    # We sum these positive differences.
    for i in range(N - rem, N):
         # The condition `if A[i] > k + 1` ensures we only sum positive differences (excesses).
         # This is equivalent to summing max(0, A[i] - (k+1)).
         if A[i] > k + 1:
             total_ops += A[i] - (k + 1)

    # Print the minimum total number of operations required to standard output
    print(total_ops)

# Execute the solve function to run the program
solve()