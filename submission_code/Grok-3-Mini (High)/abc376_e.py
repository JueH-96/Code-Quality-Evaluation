import sys
import heapq

data = sys.stdin.read().split()
index = 0
T = int(data[index])
index += 1
for test_case in range(T):
    N = int(data[index])
    K = int(data[index + 1])
    index += 2
    A_list = list(map(int, data[index:index + N]))
    index += N
    B_list = list(map(int, data[index:index + N]))
    index += N
    
    # Sort the elements by A ascending
    sorted_items = sorted(zip(A_list, B_list))  # list of (A, B) sorted by A
    
    min_ans = float('inf')
    sum_B_curr = 0
    heap_B = []  # min-heap for -B, store (-B, idx)
    heap_A = []  # min-heap for -A, store (-A, idx)
    valid_set = set()  # set of idx in K-smallest
    
    for i in range(N):  # i is index in sorted_items
        A_val, B_val = sorted_items[i]
        
        # Add the new element to the K-smallest set
        heapq.heappush(heap_B, (-B_val, i))  # -B, idx
        heapq.heappush(heap_A, (-A_val, i))  # -A, idx
        valid_set.add(i)
        sum_B_curr += B_val
        
        if len(heap_B) > K:
            # Remove the one with max B
            neg_B_pop, idx_pop = heapq.heappop(heap_B)
            B_pop = -neg_B_pop
            sum_B_curr -= B_pop
            valid_set.remove(idx_pop)
            # Do not remove from heap_A, just invalidate
        
        if i + 1 >= K:  # num elements added >= K
            # Clean heap_A
            while heap_A:
                neg_A_top, idx_top = heap_A[0]
                if idx_top not in valid_set:
                    heapq.heappop(heap_A)  # discard invalid
                else:
                    break  # top is valid
            
            # Now top of heap_A is valid, get max A
            neg_A_top, idx_top = heap_A[0]
            max_A_val = -neg_A_top  # actual A
            
            product = max_A_val * sum_B_curr
            if product < min_ans:
                min_ans = product
    
    # Output the answer for this test case
    print(int(min_ans))