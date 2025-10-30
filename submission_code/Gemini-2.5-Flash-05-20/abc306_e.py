import heapq
import collections
import sys

# Main function to encapsulate the solution logic
def solve():
    # Read N, K, Q from standard input
    N, K, Q = map(int, sys.stdin.readline().split())

    # current_A_values: Stores the current values of sequence A
    # A_1, ..., A_N corresponds to 0-indexed A[0]...A[N-1]
    current_A_values = [0] * N

    # min_heap_top_k: A min-heap to store the K largest elements.
    # The smallest element in this heap is the K-th largest overall.
    min_heap_top_k = [] 
    # max_heap_others: A max-heap to store the (N-K) smaller elements.
    # Stores negated values to simulate a max-heap with Python's min-heap.
    max_heap_others = [] 

    # Lazy deletion counters:
    # These dictionaries keep track of elements that have been conceptually
    # removed from a heap but are still physically present.
    to_be_removed_from_top_k = collections.Counter()
    to_be_removed_from_others = collections.Counter()

    # current_top_k_sum: Stores the sum of the K largest elements.
    current_top_k_sum = 0

    # Initialize heaps with N zeros.
    # Since all initial A values are 0, we distribute them into the heaps.
    # K zeros go into min_heap_top_k, and (N-K) zeros go into max_heap_others.
    # The sum of top K elements remains 0.
    for _ in range(K):
        heapq.heappush(min_heap_top_k, 0)

    for _ in range(N - K):
        heapq.heappush(max_heap_others, -0) # Store 0 as -0

    # Helper function to clean heaps:
    # Removes elements marked for removal from the heap's top.
    # It repeatedly pops the top element if it's found in the removed_counter
    # and decrements its count in the counter.
    def _clean_heap(heap, removed_counter):
        while heap and heap[0] in removed_counter and removed_counter[heap[0]] > 0:
            val = heapq.heappop(heap)
            removed_counter[val] -= 1

    # Main balancing logic:
    # This function ensures that:
    # 1. min_heap_top_k contains exactly K *active* (not removed) elements.
    # 2. Every element in min_heap_top_k is greater than or equal to
    #    every active element in max_heap_others.
    def _balance_heaps():
        nonlocal current_top_k_sum

        # Ensure heaps are clean at their tops before starting balancing logic.
        _clean_heap(min_heap_top_k, to_be_removed_from_top_k)
        _clean_heap(max_heap_others, to_be_removed_from_others)
        
        # Phase 1: Move elements from max_heap_others to min_heap_top_k
        # This happens if min_heap_top_k has fewer than K active elements
        # and there are elements in max_heap_others to promote.
        while (len(min_heap_top_k) - sum(to_be_removed_from_top_k.values()) < K and 
               max_heap_others):
            _clean_heap(max_heap_others, to_be_removed_from_others)
            if not max_heap_others: break # No more active elements to move

            val = -heapq.heappop(max_heap_others) # Get largest from max_heap_others
            heapq.heappush(min_heap_top_k, val)
            current_top_k_sum += val
            _clean_heap(min_heap_top_k, to_be_removed_from_top_k) # Clean after push

        # Phase 2: Move elements from min_heap_top_k to max_heap_others
        # This happens if min_heap_top_k has more than K active elements.
        while len(min_heap_top_k) - sum(to_be_removed_from_top_k.values()) > K:
            _clean_heap(min_heap_top_k, to_be_removed_from_top_k)
            if not min_heap_top_k: break # Should not happen if condition is met

            val = heapq.heappop(min_heap_top_k) # Get smallest from min_heap_top_k
            heapq.heappush(max_heap_others, -val)
            current_top_k_sum -= val
            _clean_heap(max_heap_others, to_be_removed_from_others) # Clean after push
        
        # Phase 3: Ensure min_heap_top_k.min() >= -max_heap_others.max()
        # If the smallest in top_k is less than the largest in others, swap them.
        _clean_heap(min_heap_top_k, to_be_removed_from_top_k)
        _clean_heap(max_heap_others, to_be_removed_from_others)

        while (min_heap_top_k and max_heap_others and 
               min_heap_top_k[0] < -max_heap_others[0]):
            
            val_from_top = heapq.heappop(min_heap_top_k)
            val_from_others = -heapq.heappop(max_heap_others)

            heapq.heappush(max_heap_others, -val_from_top)
            heapq.heappush(min_heap_top_k, val_from_others)

            current_top_k_sum = current_top_k_sum - val_from_top + val_from_others
            
            # Re-clean after swaps, as heap top might be a newly exposed removed element
            _clean_heap(min_heap_top_k, to_be_removed_from_top_k)
            _clean_heap(max_heap_others, to_be_removed_from_others)

    # Process Q updates
    for _ in range(Q):
        x_idx, new_val = map(int, sys.stdin.readline().split())
        idx = x_idx - 1 # Convert to 0-indexed

        old_val = current_A_values[idx]

        # Step 1: Mark old_val for removal from its conceptual heap.
        # Determine if old_val was among the K largest.
        # This check requires min_heap_top_k to be clean at its top.
        _clean_heap(min_heap_top_k, to_be_removed_from_top_k)
        
        is_old_val_in_top_k = False
        # If K=0, min_heap_top_k is empty, so no element can be in it.
        # current_top_k_sum will always be 0.
        if K > 0 and min_heap_top_k and old_val >= min_heap_top_k[0]:
            is_old_val_in_top_k = True
            
        if is_old_val_in_top_k:
            to_be_removed_from_top_k[old_val] += 1
            current_top_k_sum -= old_val
        else:
            to_be_removed_from_others[old_val] += 1

        # Step 2: Add new_val to the system.
        # It's generally robust to add new elements to the 'smaller' heap (max_heap_others),
        # and let the balancing logic promote it if it's large enough.
        heapq.heappush(max_heap_others, -new_val)

        # Update the actual value in the A array for future reference.
        current_A_values[idx] = new_val

        # Step 3: Rebalance heaps to maintain invariants.
        _balance_heaps()

        # Print the current sum of the K largest elements.
        sys.stdout.write(str(current_top_k_sum) + '
')

solve()