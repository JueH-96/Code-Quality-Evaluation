import heapq
import sys

def solve():
    N, K, Q = map(int, sys.stdin.readline().split())

    # A_values stores the current values of the sequence elements
    A_values = [0] * N 
    # in_topK[i] is True if A_values[i] is conceptually in the set of K largest elements (S_topK),
    # False if A_values[i] is conceptually in the set of N-K "rest" elements (S_rest).
    in_topK = [False] * N 
    
    # topK_heap is a min-heap storing (value, index) for elements conceptually in S_topK.
    topK_heap = []
    # rest_heap is a max-heap storing (-value, index) for elements conceptually in S_rest.
    # (Python's heapq is min-heap, so negative values are stored to simulate max-heap behavior)
    rest_heap = []

    current_sum_topK = 0
    
    # Initialization:
    # All elements are initially 0.
    # Arbitrarily, first K indices (0 to K-1) go to S_topK, rest (K to N-1) go to S_rest.
    # This is a valid partition since all values are 0, satisfying min(S_topK) >= max(S_rest).
    # current_sum_topK is 0.
    for i in range(K):
        heapq.heappush(topK_heap, (0, i))
        in_topK[i] = True
    
    # For rest_heap, store (-value, index). Since value is 0, -0 is 0.
    for i in range(K, N):
        heapq.heappush(rest_heap, (0, i)) 
        # in_topK[i] is already False by Python's default list initialization [False]*N

    results = [] # To store answers for each query

    for _ in range(Q):
        X, Y = map(int, sys.stdin.readline().split())
        X -= 1 # Convert to 0-indexed

        old_val = A_values[X]
        A_values[X] = Y # Update the actual value in array A_values

        if in_topK[X]:
            # This element was conceptually in S_topK.
            current_sum_topK -= old_val
            current_sum_topK += Y
            # Add the new state (Y, X) to topK_heap.
            # The old entry (old_val, X) that might be in topK_heap becomes "stale".
            heapq.heappush(topK_heap, (Y, X))
        else:
            # This element was conceptually in S_rest.
            # Add the new state (-Y, X) to rest_heap.
            # The old entry (-old_val, X) that might be in rest_heap becomes "stale".
            heapq.heappush(rest_heap, (-Y, X))

        # Rebalance heaps:
        # The `in_topK` array determines true set membership.
        # The number of elements `i` for which `in_topK[i]` is true is always K.
        # We need to ensure that min_val_in_S_topK >= max_val_in_S_rest.
        # Stale entries in heaps (value in heap != A_values[index] OR 
        # in_topK[index] mismatch) must be cleared from heap tops.
        
        while True:
            # Find smallest valid element in topK_heap
            smallest_in_topK_val = float('inf') 
            smallest_in_topK_idx = -1
            
            while topK_heap:
                val, idx = topK_heap[0] # Peek
                # Condition 1 for staleness: Element idx conceptually moved to S_rest
                if not in_topK[idx]: 
                    heapq.heappop(topK_heap)
                    continue
                # Condition 2 for staleness: Value of A_values[idx] changed
                if A_values[idx] != val: 
                    heapq.heappop(topK_heap)
                    continue
                # If we reach here, (val, idx) is a valid representation of an element in S_topK
                smallest_in_topK_val = val
                smallest_in_topK_idx = idx
                break 
            
            # Find largest valid element in rest_heap
            largest_in_rest_val = float('-inf')
            largest_in_rest_idx = -1

            while rest_heap:
                neg_val, idx = rest_heap[0] # Peek. neg_val is the stored -true_value.
                val = -neg_val # Actual value of the element
                # Condition 1 for staleness: Element idx conceptually moved to S_topK
                if in_topK[idx]: 
                    heapq.heappop(rest_heap)
                    continue
                # Condition 2 for staleness: Value of A_values[idx] changed
                if A_values[idx] != val: 
                    heapq.heappop(rest_heap)
                    continue
                # If we reach here, (val, idx) is a valid representation of an element in S_rest
                largest_in_rest_val = val
                largest_in_rest_idx = idx
                break 
            
            # Termination conditions for the balancing loop:
            # 1. If either heap is exhausted of valid elements.
            #    (This covers K=N where S_rest is empty, or K=0 if allowed, or if many stale items were cleared)
            if smallest_in_topK_idx == -1 or largest_in_rest_idx == -1:
                break 
            # 2. If the K-th largest property holds: min of S_topK >= max of S_rest
            if smallest_in_topK_val >= largest_in_rest_val:
                break

            # Property violated: smallest_in_topK_val < largest_in_rest_val.
            # Swap these two elements between the conceptual sets S_topK and S_rest.
            
            heapq.heappop(topK_heap) # Remove (smallest_in_topK_val, smallest_in_topK_idx)
            heapq.heappop(rest_heap) # Remove (-largest_in_rest_val, largest_in_rest_idx)

            current_sum_topK -= smallest_in_topK_val
            current_sum_topK += largest_in_rest_val
            
            in_topK[smallest_in_topK_idx] = False 
            in_topK[largest_in_rest_idx] = True   
            
            heapq.heappush(topK_heap, (largest_in_rest_val, largest_in_rest_idx))
            heapq.heappush(rest_heap, (-smallest_in_topK_val, smallest_in_topK_idx))
            
        results.append(current_sum_topK)

    sys.stdout.write('
'.join(map(str, results)) + '
')

solve()