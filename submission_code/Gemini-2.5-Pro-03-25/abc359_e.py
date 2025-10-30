# YOUR CODE HERE
import sys

def solve():
    # Read the integer N, the length of the sequence H.
    N = int(sys.stdin.readline())
    
    # Read the sequence H of N positive integers.
    H = list(map(int, sys.stdin.readline().split()))

    # Initialize an empty stack. The stack will store tuples representing blocks of elements.
    # Each tuple will be of the form (height, width).
    # 'height' corresponds to the value H[i] from the input sequence.
    # 'width' represents the number of consecutive elements that this block effectively covers
    # due to the monotonic stack processing logic.
    stack = [] 
    
    # Initialize the cumulative number of operations to 0.
    # This variable, `current_ops_sum`, will track the total operations needed up to the current index i,
    # considering the height constraints H encountered so far. It corresponds to K_{i+1}.
    current_ops_sum = 0
    
    # Initialize a list `ans` of size N to store the computed answers K_1, K_2, ..., K_N.
    ans = [0] * N

    # Iterate through the input heights H from index 0 to N-1.
    # The index `i` corresponds to H_{i+1} in the 1-based indexing used in the problem statement.
    for i in range(N):
        # Get the current height constraint H[i].
        h = H[i]
        
        # Initialize the width `w` for the current element `i` to 1.
        w = 1
        
        # Process the stack to maintain a monotonically increasing sequence of heights.
        # While the stack is not empty and the height of the block at the top of the stack (`stack[-1][0]`)
        # is greater than or equal to the current height `h`:
        while stack and stack[-1][0] >= h:
            # Pop the block (prev_h, prev_w) from the top of the stack.
            prev_h, prev_w = stack.pop()
            
            # Subtract the operational cost contributed by the popped block from `current_ops_sum`.
            # The cost associated with a block (prev_h, prev_w) is calculated as (prev_h + 1) * prev_w.
            # The `+1` comes from the condition A_{j-1} > H_j required for a unit to move.
            # This subtraction step effectively removes the influence of the taller block `prev_h`
            # because the current lower height `h` becomes the new bottleneck for these elements.
            current_ops_sum -= (prev_h + 1) * prev_w
            
            # Merge the width: Add the width of the popped block (`prev_w`) to the current width `w`.
            # This means the current element `i` and the elements represented by the popped block(s)
            # now form a single larger block governed by the height constraint `h`.
            w += prev_w 
        
        # Add the operational cost contributed by the new block (or the merged block).
        # This block has height `h` and total accumulated width `w`.
        # The cost added is (h + 1) * w. This represents the operations required due to the
        # height constraint `h` over `w` elements.
        current_ops_sum += (h + 1) * w
        
        # Push the new block state (h, w) onto the stack. The stack now correctly reflects
        # the state after processing element `i`.
        stack.append((h, w))
        
        # The answer K_{i+1} (corresponding to the 0-based index `i`) is the current value of `current_ops_sum`.
        # Store this value in the `ans` list at index `i`.
        ans[i] = current_ops_sum

    # Print the final answers K_1 through K_N, separated by spaces.
    # The `*ans` syntax unpacks the list `ans` into individual arguments for the print function.
    print(*(ans))

# Execute the solve function to run the program.
solve()