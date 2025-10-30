import sys

def solve():
    """
    Reads input, solves the problem, and prints the output.
    """
    try:
        # Fast I/O
        readline = sys.stdin.readline
        
        # Read N
        N = int(readline())
        
        # Read permutations P and A (1-based), convert to 0-based
        P = [int(x) - 1 for x in readline().split()]
        A = [int(x) - 1 for x in readline().split()]

    except (IOError, ValueError):
        # Handles cases with empty input, e.g., at the end of a file.
        return

    # 1. Compute the inverse of P, let's call it Q.
    # The operation `new_A[i] = A[P[i]]` means the value at index `j` moves to `Q[j]`.
    # Values on a cycle of Q are cyclically shifted among the indices of that cycle.
    Q = [0] * N
    for i in range(N):
        Q[P[i]] = i

    # `visited` array to keep track of processed indices.
    visited = [False] * N
    # `result` array to store the final permutation.
    result = [-1] * N
    
    # 2. Iterate through indices from 0 to N-1 for the greedy strategy.
    for i in range(N):
        if visited[i]:
            continue

        # If we reach here, `i` is the smallest index in an unprocessed cycle.
        
        # a. Trace the cycle starting from `i` using Q.
        cycle_q_order = []
        curr = i
        while not visited[curr]:
            visited[curr] = True
            cycle_q_order.append(curr)
            curr = Q[curr]
            
        # b. Collect the initial values from A at these cycle indices.
        cycle_values = [A[idx] for idx in cycle_q_order]
        
        # c. Greedy choice: For the smallest index `i` in this cycle,
        # assign the smallest possible value from `cycle_values`.
        # `i` is `cycle_q_order[0]` because we start the trace from the smallest index `i`.
        
        # Find the minimum value in the cycle's values and its position.
        min_val = min(cycle_values)
        min_val_idx = cycle_values.index(min_val)
        
        # d. To assign `min_val` to `result[i]`, we need to cyclically shift
        # `cycle_values` so that `min_val` moves to the first position.
        shifted_values = cycle_values[min_val_idx:] + cycle_values[:min_val_idx]
        
        # e. Assign these shifted values to the corresponding indices in the result.
        for k in range(len(cycle_q_order)):
            result[cycle_q_order[k]] = shifted_values[k]

    # Convert the 0-based result back to 1-based and print.
    print(*(x + 1 for x in result))

solve()