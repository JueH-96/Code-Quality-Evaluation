import heapq
import sys

def solve():
    N, M, L = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    b = list(map(int, sys.stdin.readline().split()))

    # Store original indices along with prices for sorting
    # Use 1-based indexing for original indices to match problem specification for forbidden pairs
    # main_dishes: list of (price, original_1_based_index) tuples, sorted by price descending
    main_dishes = sorted([(a[i], i + 1) for i in range(N)], key=lambda x: x[0], reverse=True)
    # side_dishes: list of (price, original_1_based_index) tuples, sorted by price descending
    side_dishes = sorted([(b[i], i + 1) for i in range(M)], key=lambda x: x[0], reverse=True)

    # Store forbidden pairs in a set for O(1) average time lookups
    forbidden_pairs = set()
    for _ in range(L):
        c, d = map(int, sys.stdin.readline().split())
        forbidden_pairs.add((c, d))

    # Max-heap to store (-(price_sum), main_dish_sorted_idx, side_dish_sorted_idx)
    # Using negative sum simulates a max-heap with Python's min-heap.
    pq = []
    
    # Set to keep track of visited (main_dish_sorted_idx, side_dish_sorted_idx) pairs
    # This prevents processing the same combination multiple times.
    visited_sorted_indices = set()

    # Add the most expensive combination to the heap initially.
    # Constraints guarantee N, M >= 1, so main_dishes[0] and side_dishes[0] are safe to access.
    initial_sum = main_dishes[0][0] + side_dishes[0][0]
    heapq.heappush(pq, (-initial_sum, 0, 0))
    visited_sorted_indices.add((0, 0))

    max_price = 0 # Will be updated to the highest allowed price

    while pq:
        # Pop the combination with the largest sum (due to negative storage)
        neg_current_sum, i, j = heapq.heappop(pq)
        current_sum = -neg_current_sum

        # Get the original 1-based indices to check against forbidden_pairs
        original_main_idx = main_dishes[i][1]
        original_side_idx = side_dishes[j][1]

        # Check if this combination is forbidden
        if (original_main_idx, original_side_idx) not in forbidden_pairs:
            # If not forbidden, this is the maximum possible price
            # because we are exploring sums in decreasing order.
            max_price = current_sum
            break # Found the answer, exit the loop
        
        # If the current combination IS forbidden, explore its neighbors (next best options)
        # Explore combination with the next most expensive main dish (same side dish)
        if i + 1 < N and (i + 1, j) not in visited_sorted_indices:
            next_sum = main_dishes[i + 1][0] + side_dishes[j][0]
            heapq.heappush(pq, (-next_sum, i + 1, j))
            visited_sorted_indices.add((i + 1, j))
        
        # Explore combination with the next most expensive side dish (same main dish)
        if j + 1 < M and (i, j + 1) not in visited_sorted_indices:
            next_sum = main_dishes[i][0] + side_dishes[j + 1][0]
            heapq.heappush(pq, (-next_sum, i, j + 1))
            visited_sorted_indices.add((i, j + 1))

    # Print the result to standard output
    sys.stdout.write(str(max_price) + "
")

# Call the solve function to run the program
solve()