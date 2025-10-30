import sys
import heapq

def solve():
    N, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # Create a list of (A_i, B_i) tuples
    items = []
    for i in range(N):
        items.append((A[i], B[i]))
    
    # Sort the items primarily by A_i in ascending order.
    # If A_i values are equal, the default tie-breaking order (by B_i) is acceptable,
    # as the heap will correctly select the K smallest B_values regardless.
    items.sort() 

    min_total_value = float('inf')
    
    # A max-priority queue to keep track of the K smallest B_values.
    # Python's heapq is a min-heap, so we store negative B_values to simulate a max-heap.
    # The smallest element in the min-heap (largest negative value) corresponds to the
    # largest B_value among the elements currently in the heap. This allows us to
    # efficiently remove the largest B_value if the heap size exceeds K.
    pq = [] # This will store -B_val
    current_sum_B = 0 # This will store the sum of actual B_values (positive)

    for A_val, B_val in items:
        # Add the current B_val to the heap (as negative) and update the current sum
        heapq.heappush(pq, -B_val)
        current_sum_B += B_val

        # If the heap size exceeds K, it means we have more than K elements.
        # We need to remove the largest B_val among the current set to keep only the K smallest.
        if len(pq) > K:
            # heapq.heappop(pq) removes the smallest negative value, which corresponds
            # to the largest actual B_value. We negate it to get the positive B_value.
            largest_b_in_pq = -heapq.heappop(pq) 
            current_sum_B -= largest_b_in_pq
        
        # Once the heap contains exactly K elements, we have a valid candidate set S.
        # A_val (from the current item being processed) is guaranteed to be the maximum A_i
        # within this set of K elements. This is because 'items' is sorted by A_i, so A_val
        # is the highest A encountered so far among all elements from items[0] to items[k].
        # All elements selected into the heap must have an A_value less than or equal to A_val.
        if len(pq) == K:
            min_total_value = min(min_total_value, A_val * current_sum_B)
            
    sys.stdout.write(str(min_total_value) + '
')

# Read the number of test cases
num_test_cases = int(sys.stdin.readline())

# Process each test case
for _ in range(num_test_cases):
    solve()