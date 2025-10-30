# YOUR CODE HERE
import sys

# Set a higher recursion depth limit for safety, although this solution is iterative.
# This might be useful in competitive programming environments where default limits can be restrictive.
# sys.setrecursionlimit(200000 + 10) 

def solve():
    """
    Solves the problem based on the described logic. Reads input from stdin,
    computes the counts for each building, and prints the result to stdout.
    """
    N = int(sys.stdin.readline())
    
    # Handle edge case N=0, although constraints state N >= 1.
    # If N=0, there are no buildings, so print an empty line.
    if N == 0:
        print()
        return
        
    # Handle edge case N=1.
    # If N=1, there's only one building. It has no buildings to its right,
    # so the count is 0. We must still read the height value from input.
    if N == 1:
        sys.stdin.readline() # Consume the height input line
        print(0)
        return
        
    # Read the heights of the N buildings.
    H = list(map(int, sys.stdin.readline().split()))

    # Step 1: Compute P[j] for all j = 0..N-1
    # P[j] stores the index of the *nearest* building to the left of building j
    # that is *strictly taller* than building j.
    # If no such building exists (all buildings to the left are shorter or equal), P[j] = -1.
    # We use a monotonic stack approach for O(N) computation.
    P = [-1] * N
    stack = [] # Stores indices of buildings, maintaining decreasing height order from bottom to top.

    for j in range(N):
        # While the stack is not empty and the building at the top of the stack
        # is shorter than or equal to the current building j:
        # Pop it. It cannot be the "Previous Greater Element" (PGE) for building j
        # or any subsequent building to the right of j.
        # Note: Since all heights H_i are distinct as per constraints, H[stack[-1]] <= H[j] 
        # is equivalent to H[stack[-1]] < H[j].
        while stack and H[stack[-1]] <= H[j]:
            stack.pop()
        
        # If the stack is not empty after popping, the index at the top
        # corresponds to the nearest building to the left that is taller than building j.
        # This is P[j].
        if stack:
            P[j] = stack[-1]
        
        # Push the current building index j onto the stack.
        # It maintains the stack property and might serve as the PGE for future buildings.
        stack.append(j)

    # Step 2: Compute Count[k]
    # Count[k] represents the number of indices j such that k < j <= N-1 and P[j] = k.
    # Essentially, it counts how many buildings j to the right of building k
    # have building k as their nearest taller building to the left (PGE).
    # This is computed in O(N) by iterating through all j and incrementing Count[P[j]].
    Count = [0] * N
    for j in range(N):
        # If building j has a PGE (P[j] is not -1)
        if P[j] != -1:
            # Increment the count associated with the index of that PGE.
            Count[P[j]] += 1

    # Step 3: Compute c[k] using the derived recurrence relation
    # c[k] is the answer for building k, i.e., the count of buildings j > k satisfying the condition.
    # The recurrence relation derived is: c[k] = ([P[k+1] <= k] ? 1 : 0) + c[k+1] - Count[k+1]
    # This relation allows computing c[k] based on the already computed c[k+1].
    # We compute this by iterating backwards from k = N-2 down to 0.
    # The base case is c[N-1] = 0, as building N-1 has no buildings to its right.
    # This base case is implicitly handled because we initialize the array `c` with zeros.
    c = [0] * N
    
    # Iterate k from N-2 down to 0 to compute c[k] values.
    for k in range(N - 2, -1, -1):
        # Get the PGE index for the building immediately to the right of k (building k+1).
        p_next = P[k+1] 
        
        # The condition for pair (k, j) is equivalent to P[j] <= k.
        # Check this condition specifically for j = k+1. This tells if building k+1 satisfies the condition w.r.t building k.
        # `check` is 1 if the condition holds for j=k+1, 0 otherwise.
        check = 1 if p_next <= k else 0
        
        # Apply the recurrence relation:
        # c[k] starts with the value derived from its right neighbor c[k+1].
        # It adds 1 if pair (k, k+1) satisfies the condition (`check`).
        # It subtracts Count[k+1] which corrects for double counting or structural dependencies captured by the P values and the recurrence.
        # Effectively, c[k] sums contributions from j=k+1 and all j > k+1.
        c[k] = check + c[k+1] - Count[k+1]

    # Step 4: Print the results
    # The list `c` now contains the computed counts c_0, c_1, ..., c_{N-1}.
    # Print these values separated by spaces, as required by the output format.
    print(*(c))

# Execute the solve function to run the main logic of the program.
solve()