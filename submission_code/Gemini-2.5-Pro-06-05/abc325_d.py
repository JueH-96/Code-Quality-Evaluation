import sys
import heapq

def solve():
    """
    Solves the Keyence printer scheduling problem.
    """
    try:
        N = int(sys.stdin.readline())
    except (IOError, ValueError):
        # Handles empty input
        N = 0

    if N == 0:
        print(0)
        return

    products = []
    for _ in range(N):
        T, D = map(int, sys.stdin.readline().split())
        # Store products as (start_time, end_time) tuples.
        products.append((T, T + D))

    # Sort products by their start time. This allows processing them chronologically.
    products.sort()

    # `available_products_pq` is a min-priority queue storing the end times of products
    # that are currently available for printing.
    available_products_pq = []
    
    product_idx = 0
    # `current_time` tracks the earliest time the printer is available.
    current_time = 0
    printed_count = 0

    # The simulation loop continues as long as there are unprocessed products
    # or products still waiting in the priority queue.
    while product_idx < N or available_products_pq:
        
        # If the queue is empty, the printer is idle.
        # Fast-forward time to the start of the next available product.
        if not available_products_pq:
            # This access is safe because if the queue is empty, the loop
            # continues only if product_idx < N.
            current_time = products[product_idx][0]

        # Add all products that have started by `current_time` into the priority queue.
        while product_idx < N and products[product_idx][0] <= current_time:
            _, end_time = products[product_idx]
            heapq.heappush(available_products_pq, end_time)
            product_idx += 1

        # Clean the queue: remove products that have already expired (end_time < current_time).
        while available_products_pq and available_products_pq[0] < current_time:
            heapq.heappop(available_products_pq)
            
        # If there's a printable product, choose the one with the earliest end time.
        if available_products_pq:
            # "Print" the product by removing it from the candidate pool.
            heapq.heappop(available_products_pq)
            printed_count += 1
            # The printer has a 1 microsecond cooldown.
            current_time += 1

    print(printed_count)

solve()