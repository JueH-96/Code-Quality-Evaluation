# YOUR CODE HERE
import sys

def solve():
    # Read N (number of cells) and M (number of cells initially with stones) from standard input
    N, M = map(int, sys.stdin.readline().split())
    
    # Read initial stone positions X_i
    # Constraints state 1 <= M <= 2*10^5, ensuring M is at least 1.
    # The input lines for X and A will always be present and contain M integers.
    X = list(map(int, sys.stdin.readline().split()))
    
    # Read initial stone counts A_i
    A = list(map(int, sys.stdin.readline().split()))

    # Combine X_i and A_i into pairs (position, count)
    pairs = []
    for i in range(M):
        # pairs will store tuples (X_i, A_i)
        pairs.append((X[i], A[i]))

    # Sort the pairs based on the cell index X_i.
    # After sorting, pairs[i] contains the data for the (i+1)-th occupied cell from the left.
    # Let the sorted pairs be (X_{(1)}, A_{(1)}), ..., (X_{(M)}, A_{(M)}).
    # In 0-based Python indexing, pairs[i] is (X_{(i+1)}, A_{(i+1)}).
    pairs.sort()

    # Condition 1: Check if the total number of stones equals N.
    # The final state requires exactly one stone in each of the N cells.
    # Since operations only move stones, the total number of stones must be N initially.
    total_stones = 0
    # Calculate the sum of all initial stone counts A_i.
    for i in range(M):
        # pairs[i][1] is the stone count A_{(i+1)}
        total_stones += pairs[i][1]
    
    # If the total number of stones is not equal to N, it's impossible to reach the target state.
    if total_stones != N:
        print("-1")
        return

    # Condition 2: Check if the first occupied cell is cell 1 (X_{(1)} == 1).
    # The target state requires cell 1 to have one stone.
    # Since stones can only move from cell i to i+1 (rightwards), a stone for cell 1 must either start at cell 1
    # or move into cell 1 from the left, which is impossible.
    # Therefore, cell 1 must initially contain stones if N >= 1. Specifically, X_{(1)} must be 1.
    # This is also derived from the general condition P_k >= k for k=1, where P_k is the prefix sum of stones.
    # P_1 is the number of stones initially in cell 1. We need P_1 >= 1.
    # If X_{(1)} > 1, then P_1 = 0, violating the condition. Thus, X_{(1)} = 1 is necessary.
    
    # We check pairs[0][0] which corresponds to X_{(1)}. M >= 1 is guaranteed by constraints.
    if pairs[0][0] != 1:
        print("-1")
        return
        
    # Condition 3: Check prefix sum conditions P_k >= k for all k=1..N.
    # It is known that this condition is necessary and sufficient (along with total stones = N).
    # The critical points to check are k = X_{(i)} - 1 for i=2...M.
    # This condition translates to S_{i-1} >= X_{(i)} - 1, where S_{i-1} is the sum of stones in the first i-1 occupied cells (sorted order).
    # S_{i-1} = A_{(1)} + ... + A_{(i-1)}.
    
    current_S = 0  # Tracks the prefix sum S_i = sum_{j=1}^i A_{(j)}
    initial_pos_sum = 0 # Tracks the weighted sum \sum_{i=1}^M A_i * X_i which represents the sum of initial positions of all stones.

    # Iterate through the sorted pairs (0-based index i corresponds to 1-based math index i+1)
    for i in range(M):
        # Get current pair details: x = X_{(i+1)}, a = A_{(i+1)} (1-based notation)
        # x = pairs[i][0], a = pairs[i][1] (0-based notation)
        x = pairs[i][0] 
        a = pairs[i][1] 
        
        # Check the prefix sum condition S_{i-1} >= X_{(i)} - 1 (1-based math indices)
        # This check applies for i=2...M, which corresponds to 0-based Python indices k=1...M-1.
        if i > 0: 
            # At the start of loop iteration `i`, `current_S` holds the sum of stone counts up to the previous pair (index i-1).
            # This corresponds to S_{i-1} in the 1-based notation.
            # `x` is the position X_{(i)} for the current pair (index `i`).
            # Check if S_{i-1} < X_{(i)} - 1. If true, the condition fails.
            if current_S < x - 1:
                 print("-1")
                 return

        # Update the running prefix sum S_i = S_{i-1} + A_{(i)}
        current_S += a
        
        # Accumulate the initial position sum component A_{(i)} * X_{(i)}
        # Python's built-in arbitrary precision integers handle potentially large values without overflow.
        initial_pos_sum += a * x

    # If the program execution reaches this point, all necessary conditions are satisfied.
    # A solution exists, and the minimum number of operations can be calculated.
    
    # Calculate the sum of final positions: 1 + 2 + ... + N.
    # This is the sum of positions if each cell 1..N has exactly one stone.
    # Use integer division // to ensure the result is an integer.
    final_pos_sum = N * (N + 1) // 2
    
    # The minimum number of operations required is the total displacement of all stones.
    # This equals Sum(final positions) - Sum(initial positions).
    operations = final_pos_sum - initial_pos_sum
    
    # Print the calculated minimum number of operations.
    print(operations)

# Call the solve function to execute the logic
solve()