# YOUR CODE HERE
import sys
import heapq

# Function to solve the problem
def solve():
    # Read N (length of sequence) and Q (number of queries) from input
    N, Q = map(int, sys.stdin.readline().split())
    # Read initial sequence A
    A = list(map(int, sys.stdin.readline().split()))

    # Create a frequency count array for numbers 0 to N. Size N+1.
    # Indices correspond to numbers 0, 1, ..., N.
    # The mex (minimum excluded value) of a sequence of length N can be at most N.
    # Therefore, we only need to track counts for values in the range [0, N].
    # Values greater than N do not affect the mex unless they replace a value <= N.
    counts = [0] * (N + 1) 

    # Populate initial counts based on the elements in the sequence A
    for x in A:
        # Only consider values within the relevant range [0, N]
        if x <= N: 
             counts[x] += 1

    # Create a min-heap to store numbers in the range [0, N] that are currently missing 
    # from the sequence A (i.e., their count is 0).
    # The smallest element in this heap will be the mex.
    missing_heap = []
    for k in range(N + 1): # Check numbers 0, 1, ..., N
        if counts[k] == 0:
           # If count is 0, the number k is missing. Add it to a list first.
           missing_heap.append(k) 
    
    # Convert the list of missing numbers into a min-heap structure.
    # heapify operation takes O(N) time complexity.
    heapq.heapify(missing_heap) 
    
    # Process Q queries one by one
    for _ in range(Q):
        # Read query: index i (1-based) and new value x
        i, x = map(int, sys.stdin.readline().split())
        # Convert index i to 0-based index for list A
        idx = i - 1 
        
        # Get the old value at index idx before updating the sequence
        v_old = A[idx]
        # Update the sequence A at index idx with the new value x
        A[idx] = x
        # Store the new value in a variable for clarity
        v_new = x 
        
        # Optimization: If the old value and new value are the same, 
        # the frequency counts do not change. We can potentially skip some work.
        # However, we still need to check the heap top for staleness (lazy removal),
        # as a previously removed element might now be the smallest missing one.
        # The check `v_old != v_new` correctly handles this; if they are equal, we bypass count updates.
        if v_old != v_new:
            # --- Update counts and heap based on the removal of the old value v_old ---
            # Check if the old value is within the relevant range [0, N]
            if v_old <= N: 
                counts[v_old] -= 1 # Decrement its count
                # If the count of v_old drops to 0, it means v_old is now missing.
                if counts[v_old] == 0: 
                    # Push v_old onto the heap since it's now a candidate for mex
                    heapq.heappush(missing_heap, v_old) # O(log N) operation

            # --- Update counts based on the addition of the new value v_new ---
            # Check if the new value is within the relevant range [0, N]
            if v_new <= N: 
                # Increment the count for v_new. 
                # If the count was 0 before incrementing, it means v_new was missing and is now present.
                # This change is implicitly handled by the count update and the lazy removal mechanism.
                # We check counts[v_new] before incrementing to see if it was 0.
                # If counts[v_new] == 0 before incrementing:
                #    This value might be on the heap. It will become 'stale'.
                #    The lazy removal step below will handle removing it if it reaches the top.
                counts[v_new] += 1 
        
        # --- Find the current mex using the heap with lazy removal ---
        # Repeatedly check the top element of the heap (smallest missing number candidate).
        # If its count is actually greater than 0, it means it's present in the sequence A,
        # so it's a "stale" entry in the heap. Pop it.
        # Continue until the heap top element has count 0 or the heap is empty.
        while missing_heap and counts[missing_heap[0]] > 0:
            heapq.heappop(missing_heap) # Pop stale element, O(log N) operation
            
        # After removing any stale elements from the top, the current minimum element 
        # in the heap (`missing_heap[0]`) is the smallest non-negative integer 
        # that is not present in the sequence A. This is the mex.
        # Since A has N elements, by the pigeonhole principle, at least one number 
        # in the range [0, N] must be missing. 
        # Therefore, the heap `missing_heap` will not be empty after removing stale elements.
        print(missing_heap[0])

# Call the solve function to execute the main logic when the script runs
solve()