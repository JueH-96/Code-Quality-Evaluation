import sys
import heapq

def solve():
    """
    This function solves the problem of finding the K-th largest value of 
    A_i*B_j + B_j*C_k + C_k*A_i among all N^3 combinations.
    """
    
    # Read input from stdin
    try:
        N, K = map(int, sys.stdin.readline().split())
        A = list(map(int, sys.stdin.readline().split()))
        B = list(map(int, sys.stdin.readline().split()))
        C = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # Gracefully handle cases with malformed or empty input
        return

    # Sort arrays in descending order to easily access the largest elements.
    A.sort(reverse=True)
    B.sort(reverse=True)
    C.sort(reverse=True)

    # Use a min-heap to find the K-th largest element. By storing negative values,
    # it functions as a max-heap.
    pq = []
    
    # A set to keep track of visited (i, j, k) tuples to avoid duplicates.
    visited = set()

    # Start with the largest possible value from indices (0, 0, 0).
    i, j, k = 0, 0, 0
    initial_value = A[i] * B[j] + B[j] * C[k] + C[k] * A[i]
    
    heapq.heappush(pq, (-initial_value, i, j, k))
    visited.add((i, j, k))
    
    # Pop K times to find the K-th largest value.
    ans = 0
    for _ in range(K):
        # Pop the element with the largest value.
        neg_val, cur_i, cur_j, cur_k = heapq.heappop(pq)
        ans = -neg_val

        # Explore neighbors by incrementing each index.
        # Candidate 1: Increment index i
        next_i, next_j, next_k = cur_i + 1, cur_j, cur_k
        if next_i < N and (next_i, next_j, next_k) not in visited:
            visited.add((next_i, next_j, next_k))
            new_val = A[next_i] * B[next_j] + B[next_j] * C[next_k] + C[next_k] * A[next_i]
            heapq.heappush(pq, (-new_val, next_i, next_j, next_k))

        # Candidate 2: Increment index j
        next_i, next_j, next_k = cur_i, cur_j + 1, cur_k
        if next_j < N and (next_i, next_j, next_k) not in visited:
            visited.add((next_i, next_j, next_k))
            new_val = A[next_i] * B[next_j] + B[next_j] * C[next_k] + C[next_k] * A[next_i]
            heapq.heappush(pq, (-new_val, next_i, next_j, next_k))
            
        # Candidate 3: Increment index k
        next_i, next_j, next_k = cur_i, cur_j, cur_k + 1
        if next_k < N and (next_i, next_j, next_k) not in visited:
            visited.add((next_i, next_j, next_k))
            new_val = A[next_i] * B[next_j] + B[next_j] * C[next_k] + C[next_k] * A[next_i]
            heapq.heappush(pq, (-new_val, next_i, next_j, next_k))
            
    print(ans)

# Execute the solution
solve()