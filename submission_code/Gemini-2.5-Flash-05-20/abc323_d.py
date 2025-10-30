import heapq
import collections
import sys

def solve():
    N = int(sys.stdin.readline())

    # slime_counts: A dictionary to store the total count of slimes for each size.
    # collections.defaultdict(int) initializes new keys with a default value of 0,
    # making it convenient to add counts to potentially non-existent sizes.
    slime_counts = collections.defaultdict(int)
    
    # min_heap: A min-priority queue (implemented using heapq) to store slime sizes.
    # This ensures we always process the smallest available slime size first,
    # which is crucial for propagating carries correctly.
    min_heap = []
    
    # seen_in_heap: A set to keep track of slime sizes that are currently in the min_heap.
    # This prevents adding duplicate sizes to the heap, which would be inefficient
    # and could lead to redundant processing.
    seen_in_heap = set()

    # Read initial slime configurations
    for _ in range(N):
        S, C = map(int, sys.stdin.readline().split())
        
        # Add the count C to the slime of size S.
        # Although problem states S_i are distinct, using += is robust.
        slime_counts[S] += C 
        
        # If this size S is not already in the heap, push it to ensure it gets processed.
        if S not in seen_in_heap:
            heapq.heappush(min_heap, S)
            seen_in_heap.add(S)

    final_total_slimes = 0

    # Process slimes from smallest size upwards
    while min_heap:
        # Get the smallest slime size that needs processing
        current_size = heapq.heappop(min_heap)
        
        # Remove it from the set of seen sizes as we are now processing it.
        seen_in_heap.remove(current_size)

        # Get the current total count for this size.
        current_count = slime_counts[current_size]
        
        # Once retrieved, set its count to 0 in slime_counts.
        # This effectively marks it as "processed" and ensures its count is accounted for.
        # Any new carries contributing to this size later would add to a fresh 0.
        # (Though, `seen_in_heap` handles most of this logic.)
        slime_counts[current_size] = 0 

        # Calculate how many slimes of this size remain and how many form new, larger slimes.
        remainder = current_count % 2
        carry_over = current_count // 2

        # If remainder is 1, one slime of this size cannot be paired and contributes
        # to the final minimum count.
        if remainder == 1:
            final_total_slimes += 1
        
        # If there are pairs, they form new slimes of double the size.
        if carry_over > 0:
            next_size = 2 * current_size
            
            # Add the carry_over count to the next_size's total.
            slime_counts[next_size] += carry_over
            
            # If this next_size is not already in the heap, add it to be processed later.
            # This is important to ensure all generated sizes are processed in order.
            if next_size not in seen_in_heap:
                heapq.heappush(min_heap, next_size)
                seen_in_heap.add(next_size)

    # Print the final minimum number of slimes.
    sys.stdout.write(str(final_total_slimes) + '
')

solve()