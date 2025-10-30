import sys
import heapq

def solve():
    N = int(sys.stdin.readline())
    
    # Store products as (start_time, end_time_inclusive)
    # E_i = T_i + D_i
    products_info = []
    for _ in range(N):
        T, D = map(int, sys.stdin.readline().split())
        products_info.append((T, T + D))

    # Sort products by start time T_i. If T_i are equal, Python's default tuple 
    # sorting will use the second element (E_i) as a tie-breaker.
    products_info.sort()

    # pq (min-priority queue) stores the end_times E_i of products that are 
    # currently available (i.e., have started by current_time) and not yet printed.
    # We prioritize printing products with earlier end_times.
    pq = [] 
    
    product_idx = 0  # Index for iterating through the sorted products_info list
    
    # current_time is the earliest time the printer can print next.
    # Initialized to 0. Since T_i >= 1, it will be advanced appropriately.
    current_time = 0 
    
    printed_count = 0

    # The loop continues as long as there are products to process from the input list
    # or products in the priority queue (candidates for printing).
    while product_idx < N or pq:
        
        # If the priority queue is empty, no product is currently a candidate for printing
        # based on the previous state of current_time. We must advance time.
        if not pq:
            # If no more products in the sorted list either, we are done.
            if product_idx == N:
                break
            # Advance current_time to be at least the start time of the next product.
            # If current_time is already past this start time, it remains as is.
            # Otherwise, it jumps to this product's start time.
            current_time = max(current_time, products_info[product_idx][0])
        
        # Add all products to the priority queue that have started by current_time.
        # These are products_info[i] where T_i <= current_time.
        # Their end_times E_i are pushed into the min-priority queue.
        while product_idx < N and products_info[product_idx][0] <= current_time:
            # products_info[product_idx] is a (start_time, end_time) tuple
            heapq.heappush(pq, products_info[product_idx][1]) # Push its end_time
            product_idx += 1
            
        # Remove products from the priority queue that have already ended by current_time.
        # These are products in PQ whose end_time E_j < current_time.
        # A product with E_j == current_time is still printable at current_time.
        while pq and pq[0] < current_time:
            heapq.heappop(pq)
            
        # If there's an available product in the priority queue,
        # print the one with the earliest end_time E_i (top of min-pq).
        if pq:
            # This product is chosen. Its start_time was <= current_time.
            # Its end_time is >= current_time.
            heapq.heappop(pq) # Effectively "prints" this product
            
            printed_count += 1
            
            # After printing, printer needs 1 microsecond to charge.
            # It becomes free at current_time + 1.
            current_time += 1 
        # If pq is empty at this point, no product could be scheduled at current_time.
        # The loop continues. If product_idx < N, the `if not pq:` block at the start
        # will potentially advance current_time. Otherwise, the loop may terminate.
            
    print(printed_count)

solve()