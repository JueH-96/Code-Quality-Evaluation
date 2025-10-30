import heapq

def solve():
    N_true, K = map(int, input().split())
    A_orig = list(map(int, input().split()))
    B_orig = list(map(int, input().split()))
    C_orig = list(map(int, input().split()))

    A_orig.sort(reverse=True)
    B_orig.sort(reverse=True)
    C_orig.sort(reverse=True)
    
    # Effective length of arrays for indexing:
    # The indices x,y,z for the K-th largest sum must satisfy x < K, y < K, z < K.
    # Also, indices must be less than N_true.
    # So, any index p must satisfy p < min(N_true, K).
    # This means we effectively need arrays of length min(N_true, K).
    # Let N_actual be this length. Indices will be 0 to N_actual-1.
    N_actual = min(N_true, K)
    # K >= 1, so N_actual >= 1. This ensures A, B, C below are non-empty.

    A = A_orig[:N_actual]
    B = B_orig[:N_actual]
    C = C_orig[:N_actual]
    # The length for bounds check is N_actual. E.g., for a child (i+1,j,k), check i+1 < N_actual.

    pq = []  # Min-heap, stores (-value, i, j, k)
    
    # Initial element: A[0]B[0] + B[0]C[0] + C[0]A[0]
    # This is safe because N_actual >= 1, so A[0], B[0], C[0] exist.
    val = A[0] * B[0] + B[0] * C[0] + C[0] * A[0]
    heapq.heappush(pq, (-val, 0, 0, 0))
    visited = set([(0,0,0)])
    
    count_popped = 0
    ans = -1

    while pq and count_popped < K:
        neg_val, i, j, k = heapq.heappop(pq)
        ans = -neg_val  # Store the actual positive value
        count_popped += 1

        if count_popped == K:
            break  # Found the K-th largest value
        
        # Explore neighbors
        # Neighbor (i+1, j, k)
        if i + 1 < N_actual:  # Check index bounds against N_actual
            if (i + 1, j, k) not in visited:
                new_val = A[i+1]*B[j] + B[j]*C[k] + C[k]*A[i+1]
                heapq.heappush(pq, (-new_val, i + 1, j, k))
                visited.add((i + 1, j, k))
        
        # Neighbor (i, j+1, k)
        if j + 1 < N_actual:
            if (i, j + 1, k) not in visited:
                new_val = A[i]*B[j+1] + B[j+1]*C[k] + C[k]*A[i]
                heapq.heappush(pq, (-new_val, i, j + 1, k))
                visited.add((i, j + 1, k))

        # Neighbor (i, j, k+1)
        if k + 1 < N_actual:
            if (i, j, k + 1) not in visited:
                new_val = A[i]*B[j] + B[j]*C[k+1] + C[k+1]*A[i]
                heapq.heappush(pq, (-new_val, i, j, k + 1))
                visited.add((i, j, k + 1))
                
    print(ans)

solve()