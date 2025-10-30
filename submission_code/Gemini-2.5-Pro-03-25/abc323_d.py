# YOUR CODE HERE
import sys
import heapq
import collections

# Function encapsulating the core logic to solve the slime synthesis problem
def solve():
    # Read the number of initial slime groups from standard input
    N = int(sys.stdin.readline())
    
    # Use collections.defaultdict(int) for storing slime counts. 
    # This automatically initializes the count to 0 for any size not yet encountered,
    # simplifying updates.
    slime_counts = collections.defaultdict(int)
    
    # Read all initial slime data (size S, count C) pairs.
    # Store them in a list first to ensure all input is read before processing begins.
    initial_data = []
    for _ in range(N):
         # Read size S_i and count C_i for each group. Use tuple for pairs.
         initial_data.append(tuple(map(int, sys.stdin.readline().split())))

    # Populate the slime_counts dictionary with the initial data.
    for S, C in initial_data:
         # The problem statement guarantees that initial sizes S_i are distinct.
         # So, we can directly assign the count C to size S.
         # If S_i could repeat, we would need to use `slime_counts[S] += C`.
         slime_counts[S] = C 
    
    # Initialize a min-heap (priority queue). This heap will store the sizes S 
    # that are currently eligible for synthesis, meaning `slime_counts[S] >= 2`.
    # Using a min-heap ensures that we always process the smallest available size first,
    # which is crucial for correctness as synthesis at size X affects size 2X.
    eligible_heap = []
    for S, C in slime_counts.items():
         # If an initial size S has a count of 2 or more, it's eligible for synthesis.
         # Add such sizes to the min-heap.
         if C >= 2:
             heapq.heappush(eligible_heap, S)

    # Main loop: continue processing as long as there are sizes eligible for synthesis in the heap.
    while eligible_heap:
        # Extract the smallest size S currently eligible for synthesis from the heap.
        S = heapq.heappop(eligible_heap)
        
        # Retrieve the current count of slimes of size S from the dictionary.
        # This is necessary because the count might have changed since S was added to the heap
        # (e.g., if this entry in the heap is stale due to an earlier processing of the same size).
        C = slime_counts[S]
        
        # Verify eligibility: If the current count C is less than 2, this size S is no longer
        # eligible for synthesis (or this heap entry was stale). Skip processing.
        if C < 2:
            continue 
            
        # Calculate how many synthesis operations can be performed using pairs of slimes of size S.
        # Each synthesis consumes two slimes.
        num_syntheses = C // 2
        
        # Calculate how many slimes of size S will remain after performing all possible syntheses.
        # This will be 0 if C was even, and 1 if C was odd.
        remaining_S = C % 2 
        
        # Update the count for size S in the dictionary to reflect the remaining slimes.
        slime_counts[S] = remaining_S 
        # After this update, slime_counts[S] is either 0 or 1. Size S is no longer eligible
        # for synthesis based on its own slimes. It can only become eligible again if slimes
        # are synthesized from size S/2, but we process smaller sizes first, so this won't happen.
        
        # Determine the target size T for the new slimes produced by synthesis.
        # Each synthesis of two size S slimes produces one size 2*S slime.
        T = 2 * S
        
        # Record the count of slimes of size T *before* adding the newly synthesized slimes.
        # This value is needed later to check if T becomes newly eligible for synthesis.
        current_T_count = slime_counts[T] # defaultdict returns 0 if T is not yet in the dict
        
        # Add the `num_syntheses` newly created slimes of size T to the count for size T.
        slime_counts[T] += num_syntheses
        
        # Get the updated count of slimes of size T after the addition.
        new_T_count = slime_counts[T]
        
        # Check if size T has become eligible for synthesis due to the added slimes.
        # T becomes newly eligible if its count is now 2 or more (new_T_count >= 2) 
        # AND its count was previously less than 2 (current_T_count < 2).
        if new_T_count >= 2 and current_T_count < 2:
             # If T just became eligible, add it to the heap to be processed in a future iteration.
             # The heap property ensures it will be processed in the correct order relative to other sizes.
             heapq.heappush(eligible_heap, T)

    # After the loop terminates, no more syntheses are possible. All sizes have counts 0 or 1.
    # Calculate the final total number of slimes remaining.
    # This is the sum of counts for all sizes present in the `slime_counts` dictionary.
    # `slime_counts.values()` returns an iterator over all counts, including those that became 0.
    # Summing these values correctly computes the total number of slimes.
    total_slimes = sum(slime_counts.values())
        
    # Print the minimum possible number of slimes to standard output.
    print(total_slimes)

# Execute the solve function to run the program based on standard input.
solve()