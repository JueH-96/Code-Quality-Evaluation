# YOUR CODE HERE
import sys

def solve():
    # Read the number of stops N
    N = int(sys.stdin.readline())
    
    # According to constraints, N >= 1.
    # Read the list of passenger changes A_1, A_2, ..., A_N
    # We read the whole line and split it into integers.
    # Using list(map(...)) converts the map object into a list.
    A = list(map(int, sys.stdin.readline().split()))

    # Initialize current_sum to track the prefix sum S_k = sum_{j=1 to k} A_j.
    # It starts at S_0 = 0.
    current_sum = 0
    
    # Initialize min_sum to track the minimum prefix sum encountered so far.
    # This includes S_0 = 0. So min_sum = min(S_0, S_1, ..., S_N).
    # Initialize with S_0 = 0.
    min_sum = 0

    # Iterate through the changes A_i (using 0-based index k for A[k])
    for k in range(N):
        # Update the prefix sum. After processing A[k], current_sum holds S_{k+1}.
        current_sum += A[k]
        
        # Update the minimum prefix sum found so far.
        # This compares the current minimum with the newly computed prefix sum S_{k+1}.
        min_sum = min(min_sum, current_sum)

    # After the loop, current_sum holds the final prefix sum S_N.
    # min_sum holds the minimum value among all prefix sums S_0, S_1, ..., S_N.

    # The number of passengers P_k at any stop k must be non-negative.
    # Let P_0 be the initial number of passengers. P_k = P_0 + S_k.
    # We need P_0 >= 0 and P_k = P_0 + S_k >= 0 for all k = 1, ..., N.
    # This implies P_0 >= -S_k for all k = 1, ..., N.
    # Combining P_0 >= 0 and P_0 >= -S_k for k=1..N, we get
    # P_0 >= max(0, -S_1, -S_2, ..., -S_N).
    # Alternatively, the condition is P_k >= 0 for k=0..N, which means P_0 + S_k >= 0 for k=0..N.
    # This implies P_0 >= -S_k for all k=0..N.
    # So, P_0 >= max(-S_0, -S_1, ..., -S_N).
    # Since S_0 = 0, -S_0 = 0. Thus max(-S_k for k=0..N) is always >= 0.
    # Let M = max(-S_k for k=0..N). The minimum possible initial P_0 is M.
    # We proved earlier that M = max(0, -S_min), where S_min = min(S_0, S_1, ..., S_N).
    # So the minimum required initial number of passengers is:
    min_initial_passengers = max(0, -min_sum)
    
    # The current number of passengers is P_N = P_0 + S_N.
    # We want to find the minimum possible value of P_N. This is achieved when P_0 is minimized.
    # Minimum possible P_N = (minimum P_0) + S_N
    # Minimum possible P_N = min_initial_passengers + current_sum (which is S_N)
    final_passengers = min_initial_passengers + current_sum
    
    # Print the minimum possible current number of passengers.
    print(final_passengers)

# Execute the solve function to run the program logic
solve()
# YOUR CODE HERE