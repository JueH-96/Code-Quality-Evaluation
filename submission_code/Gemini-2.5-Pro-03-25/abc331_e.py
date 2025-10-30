# YOUR CODE HERE
import heapq
import sys

# Potentially increase recursion depth if needed, although this algorithm is iterative.
# It might be relevant for system configurations with low default recursion limits,
# although Python's heapq and set operations are generally implemented iteratively.
# sys.setrecursionlimit(200000) 

def solve():
    # Read N (number of main dishes), M (number of side dishes), L (number of forbidden pairs) from input
    N, M, L = map(int, sys.stdin.readline().split())
    
    # Read main dish prices into list a
    a = list(map(int, sys.stdin.readline().split()))
    # Read side dish prices into list b
    b = list(map(int, sys.stdin.readline().split()))
    
    # Create lists of tuples (price, original_index_0_based) for main dishes
    # Store original index to track which pair is forbidden later
    P = []
    for i in range(N):
        P.append((a[i], i))
        
    # Create lists of tuples (price, original_index_0_based) for side dishes
    Q = []
    for i in range(M):
        Q.append((b[i], i))

    # Sort P (main dishes) based on prices in descending order
    P.sort(key=lambda x: x[0], reverse=True)
    # Sort Q (side dishes) based on prices in descending order
    Q.sort(key=lambda x: x[0], reverse=True)

    # Store forbidden pairs in a set for efficient lookup O(1) on average.
    # Convert 1-based input indices (c_i, d_i) from problem statement to 0-based indices (c_i-1, d_i-1) used internally.
    forbidden_pairs = set()
    for _ in range(L):
        c, d = map(int, sys.stdin.readline().split())
        forbidden_pairs.add((c - 1, d - 1))

    # Initialize a max heap (priority queue) using Python's heapq module, which implements a min-heap.
    # To simulate a max-heap, we store negative sums. The element with the smallest negative sum corresponds to the largest actual sum.
    # Store tuples: (-sum, k, l) where sum is the price, k is the index in sorted list P, l is the index in sorted list Q.
    pq = []
    
    # Initialize a set to keep track of visited pairs of indices (k, l) from the sorted lists.
    # This prevents adding the same state (combination of k-th main dish and l-th side dish from sorted lists) multiple times to the priority queue, avoiding redundant work and potential infinite loops.
    visited_indices = set()

    # Constraints guarantee N >= 1 and M >= 1, so P and Q are non-empty.
    # The initial candidate for the maximum price is the pair combining the most expensive main dish (P[0]) and the most expensive side dish (Q[0]).
    initial_k = 0 # Index in P
    initial_l = 0 # Index in Q
    # Calculate the sum for the initial pair (P[0], Q[0])
    initial_sum = P[initial_k][0] + Q[initial_l][0]
    
    # Push the initial state onto the priority queue. Store the negative sum.
    heapq.heappush(pq, (-initial_sum, initial_k, initial_l))
    # Mark the initial state (indices 0, 0) as visited
    visited_indices.add((initial_k, initial_l))

    # Variable to store the maximum price found for an offered (non-forbidden) set meal
    max_offered_price = 0

    # Process states from the priority queue. The loop continues as long as there are candidate pairs in the queue.
    while pq:
        # Pop the state with the highest sum (which has the smallest negative sum from the min-heap)
        neg_current_sum, k, l = heapq.heappop(pq)
        # Get the actual sum by negating the stored value
        current_sum = -neg_current_sum
        
        # Retrieve the original 0-based indices corresponding to the current state (k, l).
        # P[k][1] is the original index of the k-th most expensive main dish.
        # Q[l][1] is the original index of the l-th most expensive side dish.
        original_idx_main = P[k][1]
        original_idx_side = Q[l][1]

        # Check if the pair formed by these original indices is one of the forbidden pairs.
        if (original_idx_main, original_idx_side) in forbidden_pairs:
            # If the current pair IS forbidden, we cannot choose it. We must explore its "neighbors"
            # in the grid of potential sums to find the next highest potential price.
            
            # Neighbor 1: Consider the pair formed by using the next most expensive main dish (index k+1)
            #             with the current side dish (index l).
            next_k = k + 1
            # Check if index k+1 is valid (within the bounds of the sorted main dish list P)
            if next_k < N:
                # Check if this neighbor state (next_k, l) has already been visited/added to the queue.
                # If not visited, process it.
                if (next_k, l) not in visited_indices:
                    # Calculate the sum for this neighbor pair
                    neighbor_sum = P[next_k][0] + Q[l][0]
                    # Push the neighbor state onto the priority queue (store negative sum)
                    heapq.heappush(pq, (-neighbor_sum, next_k, l))
                    # Mark this neighbor state as visited
                    visited_indices.add((next_k, l))

            # Neighbor 2: Consider the pair formed by using the current main dish (index k)
            #             with the next most expensive side dish (index l+1).
            next_l = l + 1
            # Check if index l+1 is valid (within the bounds of the sorted side dish list Q)
            if next_l < M:
                 # Check if this neighbor state (k, next_l) has already been visited/added to the queue.
                 # If not visited, process it.
                 if (k, next_l) not in visited_indices:
                    # Calculate the sum for this neighbor pair
                    neighbor_sum = P[k][0] + Q[next_l][0]
                    # Push the neighbor state onto the priority queue (store negative sum)
                    heapq.heappush(pq, (-neighbor_sum, k, next_l))
                    # Mark this neighbor state as visited
                    visited_indices.add((k, next_l))
        else:
            # If the current pair is NOT forbidden:
            # Since we explore pairs in decreasing order of their sum price (due to using a max-heap logic),
            # the first non-forbidden pair we encounter must have the maximum possible price among all offered meals.
            max_offered_price = current_sum
            # We have found the answer, so we can stop the search.
            break 

    # Print the maximum price found. If all possible pairs are forbidden (which is ruled out by problem constraints: NM - L >= 1),
    # this would output 0 if initialized to 0. But the problem guarantees at least one offered meal.
    print(max_offered_price)

# Execute the main solve function when the script runs
solve()