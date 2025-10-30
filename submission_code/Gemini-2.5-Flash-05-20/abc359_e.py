import sys

def solve():
    # Read N from standard input
    N = int(sys.stdin.readline())
    
    # Read H values and store them in a list.
    # H_i from problem statement corresponds to H_arr[i-1].
    H_arr = list(map(int, sys.stdin.readline().split()))

    # List to store the calculated answers for A_1, A_2, ..., A_N.
    # answers[i] will store the result for A_{i+1}.
    answers = []
    
    # current_sum_of_maxes stores the sum: sum_{j=0}^{i} (max_{k=j}^{i} H_arr[k])
    # This is the dynamic component of the formula 1 + sum_of_maxes.
    current_sum_of_maxes = 0 
    
    # Monotonic stack: stores (value, count) pairs.
    # 'value' is an H_arr[k] that is currently a maximum for certain suffix segments.
    # 'count' is the number of contiguous elements (from the current position 'i' backwards)
    # for which this 'value' is the maximum.
    # The stack maintains elements in non-increasing order of 'value'.
    stack = [] 

    # Iterate through H_arr (from H_arr[0] to H_arr[N-1])
    # H_arr[i] corresponds to H_{i+1} in the problem statement,
    # and we are calculating the answer for A_{i+1}.
    for i in range(N):
        h_val = H_arr[i] # Current H value being processed (H_{i+1} from problem)
        
        # 'count' represents the length of the segment (ending at current i)
        # for which h_val is the maximum.
        # Initially, h_val itself makes up a segment of length 1 (i.e., just H_arr[i]).
        count = 1 
        
        # While the stack is not empty and the top element's value is less than current h_val:
        # This means h_val is a new, larger maximum for segments previously dominated by popped elements.
        while stack and stack[-1][0] < h_val:
            popped_val, popped_count = stack.pop()
            
            # Remove the contribution of the popped element from the total sum.
            # Its 'popped_count' segments are no longer maximised by 'popped_val'.
            current_sum_of_maxes -= popped_val * popped_count
            
            # The segments previously maximised by 'popped_val' are now maximised by 'h_val'.
            # Add these segments to 'h_val's effective count.
            count += popped_count
            
        # Add h_val's new contribution to the total sum.
        # 'h_val' is the maximum for 'count' segments (ending at current 'i').
        current_sum_of_maxes += h_val * count
        
        # Push the current h_val and its accumulated count onto the stack.
        stack.append((h_val, count))
        
        # The answer for A_{i+1} (which is answers[i] in 0-indexed)
        # is 1 (for A_{i+1} itself becoming >0) + the sum of maximums of all segments
        # from j=0 to i, ending at i. This sum is maintained in current_sum_of_maxes.
        answers.append(1 + current_sum_of_maxes)

    # Print the answers separated by spaces to standard output.
    sys.stdout.write(" ".join(map(str, answers)) + "
")

# Call the solve function to run the program
solve()