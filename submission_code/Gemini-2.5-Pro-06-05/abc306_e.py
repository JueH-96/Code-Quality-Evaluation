# YOUR CODE HERE
import sys
import heapq
from collections import Counter

def solve():
    """
    Solves the problem by maintaining two heaps for the top K and the rest of the elements,
    using lazy deletion to handle updates efficiently.
    """
    # Use fast I/O
    get_input = sys.stdin.readline

    try:
        # Read problem parameters
        line = get_input()
        if not line: return
        N, K, Q = map(int, line.split())
    except (IOError, ValueError):
        return

    # Special case: K=N. f(A) is the sum of all elements.
    if K == N:
        A = [0] * N
        total_sum = 0
        for _ in range(Q):
            X, Y = map(int, get_input().split())
            idx = X - 1
            old_val = A[idx]
            A[idx] = Y
            total_sum = total_sum - old_val + Y
            sys.stdout.write(str(total_sum) + '
')
        return
        
    # K=0 is not possible under the constraints (1 <= K). If it were, sum is always 0.
    if K == 0:
        for _ in range(Q):
            get_input()
            sys.stdout.write('0
')
        return

    # --- General Case: 1 <= K < N ---
    A = [0] * N
    
    # top_k_heap: min-heap for the K largest values
    top_k_heap = [0] * K
    # rest_heap: max-heap (using negative values) for the N-K smallest values
    rest_heap = [0] * (N - K)
    
    sum_top_k = 0
    
    # Lazy deletion counters
    deleted_top_k = Counter()
    deleted_rest = Counter()
    
    # --- Heap Helper Functions with Lazy Deletion ---
    def clean_heap(heap, deleted_map, sign=1):
        """Removes marked-as-deleted elements from the top of a heap."""
        while heap and heap[0] * sign in deleted_map and deleted_map[heap[0] * sign] > 0:
            val = heapq.heappop(heap) * sign
            deleted_map[val] -= 1
            if deleted_map[val] == 0:
                del deleted_map[val]

    def get_min_top():
        """Gets the minimum element from the top_k_heap after cleaning."""
        clean_heap(top_k_heap, deleted_top_k)
        return top_k_heap[0]

    def pop_min_top():
        """Pops the minimum element from the top_k_heap after cleaning."""
        clean_heap(top_k_heap, deleted_top_k)
        return heapq.heappop(top_k_heap)

    def pop_max_rest():
        """Pops the maximum element from the rest_heap after cleaning."""
        clean_heap(rest_heap, deleted_rest, -1)
        return -heapq.heappop(rest_heap)

    # --- Main Update Loop ---
    for _ in range(Q):
        X, Y = map(int, get_input().split())
        idx = X - 1
        
        old_val = A[idx]
        new_val = Y
        A[idx] = new_val

        # --- Step 1: Remove old_val's contribution ---
        min_top_val = get_min_top()
        
        if old_val >= min_top_val:
            # old_val was in top_k. Mark for deletion.
            deleted_top_k[old_val] += 1
            sum_top_k -= old_val
            
            # To maintain K elements in top_k, promote the largest from rest.
            promoted_val = pop_max_rest()
            heapq.heappush(top_k_heap, promoted_val)
            sum_top_k += promoted_val
        else:
            # old_val was in rest. Mark for deletion.
            deleted_rest[old_val] += 1

        # --- Step 2: Add new_val's contribution ---
        min_top_val_after_removal = get_min_top()
        
        if new_val > min_top_val_after_removal:
            # new_val belongs in top_k. It displaces the current smallest element.
            demoted_val = pop_min_top()
            sum_top_k -= demoted_val
            heapq.heappush(rest_heap, -demoted_val)
            
            # Add new_val to top_k
            heapq.heappush(top_k_heap, new_val)
            sum_top_k += new_val
        else:
            # new_val belongs in rest
            heapq.heappush(rest_heap, -new_val)
            
        sys.stdout.write(str(sum_top_k) + '
')

solve()