def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    # Parsing inputs
    N, M, L = map(int, input_data[:3])
    a_vals = list(map(int, input_data[3:3+N]))
    b_vals = list(map(int, input_data[3+N:3+N+M]))
    pairs_data = input_data[3+N+M:]
    
    # Read disallowed pairs (c_i, d_i)
    # Note: They are 1-indexed in the input
    disallowed = set()
    idx = 0
    for _ in range(L):
        c_i = int(pairs_data[idx]) - 1
        d_i = int(pairs_data[idx+1]) - 1
        idx += 2
        disallowed.add((c_i, d_i))
    
    # Sort main dishes by cost descending, keep (cost, original_index)
    A = sorted([(a_vals[i], i) for i in range(N)], key=lambda x: x[0], reverse=True)
    # Sort side dishes by cost descending, keep (cost, original_index)
    B = sorted([(b_vals[j], j) for j in range(M)], key=lambda x: x[0], reverse=True)
    
    # We use a max-heap (Python's heapq is a min-heap, so we'll store negative sums)
    import heapq
    
    # Push (-(a[0]+b[0]), i=0, j=0) onto a heap
    heap = []
    visited = set()
    heapq.heappush(heap, (-(A[0][0] + B[0][0]), 0, 0))  # (negative sum, i, j)
    visited.add((0, 0))
    
    # Pop from the heap until we find an allowed pair
    while heap:
        neg_sum, i, j = heapq.heappop(heap)
        sum_val = -neg_sum
        # Check if disallowed
        # A[i][1] -> original index of the i-th largest main dish
        # B[j][1] -> original index of the j-th largest side dish
        if (A[i][1], B[j][1]) not in disallowed:
            # Found the best (largest sum) allowed pair
            print(sum_val)
            return
        # Otherwise, push neighbors if not visited
        if i+1 < N and (i+1, j) not in visited:
            heapq.heappush(heap, (-(A[i+1][0] + B[j][0]), i+1, j))
            visited.add((i+1, j))
        if j+1 < M and (i, j+1) not in visited:
            heapq.heappush(heap, (-(A[i][0] + B[j+1][0]), i, j+1))
            visited.add((i, j+1))

# Call solve() after defining
if __name__ == "__main__":
    solve()