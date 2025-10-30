# YOUR CODE HERE
import sys

# Attempt to import SortedList from sortedcontainers library.
# This library provides an efficient implementation of a sorted list data structure,
# which behaves like a balanced binary search tree supporting O(log N) operations.
try:
    from sortedcontainers import SortedList
except ImportError:
    # If the library is not available, print an error message to stderr and exit.
    # This might happen if the code is run in an environment without the library installed.
    # For competitive programming platforms, check if this library is supported.
    # If not, an alternative implementation (e.g., using segment trees or implementing a balanced BST)
    # would be required.
    sys.stderr.write("Error: 'sortedcontainers' library not found.
")
    sys.stderr.write("Please install it using 'pip install sortedcontainers'.
")
    exit(1) 

def solve():
    # Read input values N, K, Q
    # N: length of the sequence A
    # K: number of largest elements to sum
    # Q: number of updates
    N, K, Q = map(int, sys.stdin.readline().split())

    # Initialize two SortedList structures:
    # S: stores the K largest elements currently in A.
    # T: stores the remaining N-K elements.
    # Initially, A contains N zeros.
    S = SortedList([0] * K)
    
    # Initialize T only if N > K. If N=K, T remains empty.
    if N > K:
        T = SortedList([0] * (N - K))
    else: 
        T = SortedList() # T is empty if K=N

    # Maintain the sum of elements currently in S. Initially, it's 0.
    sum_S = 0
    
    # Store the current state of the sequence A using a list.
    # Array indices are 0-based, while input indices X_i are 1-based.
    current_A = [0] * N 

    # Process Q updates
    for _ in range(Q):
        # Read update parameters X_i and Y_i
        # X_i: index in A to update (1-based)
        # Y_i: new value for A[X_i]
        X, Y = map(int, sys.stdin.readline().split())
        X -= 1 # Convert X to 0-based index for list access

        # Get the old value at index X before the update
        V = current_A[X]
        
        # Update the value at index X in our record of A
        current_A[X] = Y

        # --- Update Step ---

        # Step 1: Remove the old value V from the appropriate set (S or T).
        
        # Check if V is present in S using the efficient `in` operator of SortedList.
        if V in S:
            # If V was in S, remove one instance of it from S.
            S.remove(V) 
            # Decrease the sum of elements in S.
            sum_S -= V
            
            # Since an element was removed from S, its size decreases to K-1.
            # To restore its size to K and maintain the property that S contains the K largest elements,
            # we must promote the largest element from T (if T is not empty) to S.
            if len(T) > 0:
                # Get and remove the maximum element from T using pop(-1).
                move_val = T.pop(-1) 
                # Add this element to S.
                S.add(move_val)      
                # Update the sum of S.
                sum_S += move_val    
        else:
            # If V was not in S, it must have been in T.
            # This case is only possible if K < N, meaning T is potentially non-empty.
            if len(T) > 0:
                 # Remove one instance of V from T.
                 # This assumes V exists in T; otherwise, implies inconsistent state.
                 T.remove(V)
            # If T is empty (which happens only if K=N), then V must be in S,
            # so this 'else' branch should not be logically reachable if V existed in A.

        # Step 2: Add the new value Y into the structure.
        # We tentatively add Y to S first. This potentially increases S size temporarily.
        S.add(Y)
        # Update the sum of S accordingly.
        sum_S += Y

        # Step 3: Rebalance S and T to maintain invariants.
        # After adding Y, S might have K+1 elements. This happens if V was removed from T, 
        # and Y was added to S. Or if V was removed from S, max(T) moved to S, then Y added to S.
        # If S size exceeds K, the smallest element currently in S must be moved to T.
        if len(S) > K:
            # Get and remove the minimum element from S using pop(0).
            move_val = S.pop(0) 
            # Update the sum of S.
            sum_S -= move_val   
            # Add this element to T.
            T.add(move_val)     

        # After these operations:
        # - S contains exactly K elements.
        # - T contains exactly N-K elements.
        # - S contains the K largest elements among all N elements.
        # - sum_S correctly reflects the sum of elements in S.
        
        # Print the current sum_S, which is the value f(A) after the update.
        print(sum_S)

# Execute the main solve function to process the input and print results.
solve()