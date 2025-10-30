# YOUR CODE HERE
import sys

# This function implements the greedy strategy discussed during the thought process.
# The strategy is based on interpreting the problem as finding a minimum cost pairing
# to maximize the final sum, using the formula: Final Sum = Sum(All A_i) - Sum(Cost of Pairs).
# The cost of a pair (k, j) corresponding to deleting element A_k at step j is A_k + A_j.
# The algorithm attempts to minimize the sum of pair costs greedily.

def solve():
    # Read the number of elements N
    N = int(sys.stdin.readline())
    
    # Handle edge case N=0, although constraints state N >= 1
    if N == 0:
        print(0)
        return

    # Read the sequence A
    A = list(map(int, sys.stdin.readline().split()))

    # The stack `st` will store pairs (value, original_index) of elements currently in sequence S.
    # Using the original index is helpful for debugging/understanding but not strictly necessary for the algorithm logic.
    st = [] 
    
    # `current_sum` tracks the sum of elements currently considered to be in the sequence S.
    current_sum = 0

    # Process each element A_i from i = 0 to N-1
    for i in range(N):
        val = A[i]
        
        # Step 1: Tentatively perform the 'Append A_i' operation.
        # Push the current element onto the stack and add its value to the running sum.
        st.append((val, i)) # Use 0-based index i
        current_sum += val

        # Step 2: Check if the last operation should have been 'Delete' instead of 'Append'.
        # This check is implemented greedily based on minimizing pair sums locally.
        # If the top two elements on the stack, say A_k (second top) and A_i (top), 
        # sum to a negative value (A_k + A_i < 0), it indicates that forming the pair (k, i)
        # is beneficial under the greedy strategy. This pair corresponds to using the operation
        # at step `i` to delete A_k, instead of appending A_i.
        # The `while` loop handles potential cascading effects: after forming a pair and popping,
        # the new top pair might also sum to negative.
        while len(st) >= 2:
            # Access the top two elements
            val1, idx1 = st[-1] # Top element (value = A[i] or from previous pops)
            val2, idx2 = st[-2] # Element below the top (value = A[k])
            
            # Condition for forming the pair (k, i)
            if val1 + val2 < 0:
                 # If A_k + A_i < 0, perform the effective 'Delete' operation retroactively.
                 # Pop both elements from the stack.
                 st.pop() # Remove (val1, idx1)
                 st.pop() # Remove (val2, idx2)
                 
                 # Update the sum. Both val1 (A_i) and val2 (A_k) were added to `current_sum` previously.
                 # Subtracting their sum (val1 + val2) correctly updates `current_sum` to reflect
                 # that A_k was added then removed (net 0 contribution), and A_i was never effectively added
                 # (net 0 contribution compared to initially adding and then removing).
                 # Overall change = -val1 - val2.
                 current_sum -= (val1 + val2) 
            else:
                 # If the sum A_k + A_i >= 0, the greedy choice is not to form this pair.
                 # The current stack configuration is considered locally optimal.
                 # Stop checking for pairs involving the currently pushed element A_i.
                 break
                
    # After processing all N elements, `current_sum` holds the sum of elements
    # remaining in the sequence S according to the applied greedy strategy.
    print(current_sum)

# Execute the solve function
solve()