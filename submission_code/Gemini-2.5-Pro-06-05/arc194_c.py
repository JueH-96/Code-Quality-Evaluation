import sys

def solve():
    """
    This function reads the input, solves the problem, and prints the answer.
    """
    
    # Read input from stdin
    try:
        input_stream = sys.stdin.read
        lines = input_stream().splitlines()
        
        if not lines:
            return
            
        N = int(lines[0])
        A = list(map(int, lines[1].split()))
        B = list(map(int, lines[2].split()))
        C = list(map(int, lines[3].split()))
    except (IOError, ValueError, IndexError):
        # Handle potential empty lines or invalid format
        return

    # Calculate the initial sum S_0 = sum(A_k * C_k)
    s0 = 0
    for i in range(N):
        if A[i] == 1:
            s0 += C[i]

    # --- Case 1: Minimal flips ---
    # We only flip elements where A[i] != B[i].
    
    d_mismatch = []
    for i in range(N):
        if A[i] != B[i]:
            if A[i] == 0:  # Flip 0 -> 1
                # The change in the sum S is +C[i]
                d_mismatch.append(C[i])
            else:  # Flip 1 -> 0
                # The change in the sum S is -C[i]
                d_mismatch.append(-C[i])

    # If A and B are already identical, cost is 0.
    if not d_mismatch:
        print(0)
        return

    # Helper function to calculate total cost given a list of deltas
    def calculate_total_cost(deltas, initial_sum):
        if not deltas:
            return 0
        
        # Sort deltas to apply them in optimal order (smallest first)
        deltas.sort()
        
        total_cost = 0
        current_sum = initial_sum
        for d in deltas:
            current_sum += d
            total_cost += current_sum
        return total_cost

    cost1 = calculate_total_cost(list(d_mismatch), s0)

    # --- Case 2: Minimal flips + all possible extra flip pairs ---
    # An "extra" flip pair at index j (where A[j] == B[j]) consists of
    # flipping A[j] and then flipping it back. This adds (d_j, -d_j)
    # to the list of sum changes.
    # Cost reduction is plausible if d_j is negative, i.e., A[j]=B[j]=1.
    # The problem structure suggests it's optimal to either take no extra
    # pairs, or all such beneficial pairs.
    
    d_full = list(d_mismatch)
    has_extra_pairs = False
    for i in range(N):
        if A[i] == 1 and B[i] == 1:
            d_full.append(C[i])
            d_full.append(-C[i])
            has_extra_pairs = True
            
    if not has_extra_pairs:
        # If no A[i]=B[i]=1, then Case 2 is the same as Case 1
        print(cost1)
        return

    cost2 = calculate_total_cost(d_full, s0)
    
    print(min(cost1, cost2))

# Execute the solution
solve()