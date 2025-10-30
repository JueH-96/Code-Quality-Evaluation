def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N, M, L = map(int, input_data[:3])
    idx = 3
    
    # Read main dishes
    a = list(map(int, input_data[idx:idx+N]))
    idx += N
    
    # Read side dishes
    b = list(map(int, input_data[idx:idx+M]))
    idx += M
    
    # Read forbidden pairs
    forbidden = set()
    for _ in range(L):
        ci = int(input_data[idx]); di = int(input_data[idx+1])
        idx += 2
        # Switch to 0-based indexing
        forbidden.add((ci - 1, di - 1))
    
    # Sort main dishes in descending order (cost, original_index)
    A = sorted([(a_i, i) for i, a_i in enumerate(a)], key=lambda x: x[0], reverse=True)
    # Sort side dishes in descending order (cost, original_index)
    B = sorted([(b_j, j) for j, b_j in enumerate(b)], key=lambda x: x[0], reverse=True)
    
    import heapq
    
    # Max-heap approach in "index space" over the sorted arrays A and B.
    # We store entries as (-sum, iA, iB) so that popping the min-heap
    # gives us the largest sum (since Python's heap is a min-heap).
    
    heap = []
    visited = set()  # to avoid pushing the same (iA, iB) multiple times
    
    # Push the top (iA=0, iB=0) if both arrays are non-empty
    if N > 0 and M > 0:
        cost_top = A[0][0] + B[0][0]
        heapq.heappush(heap, (-cost_top, 0, 0))
        visited.add((0, 0))
    
    # Repeatedly pop from the heap until we find a non-forbidden pair
    while heap:
        neg_sum, iA, iB = heapq.heappop(heap)
        current_sum = -neg_sum
        
        # Get the original indices
        main_idx = A[iA][1]
        side_idx = B[iB][1]
        
        # Check forbidden
        if (main_idx, side_idx) not in forbidden:
            print(current_sum)
            return
        
        # If forbidden, push neighbors (iA+1, iB) and (iA, iB+1) if in range
        if iA + 1 < N:
            if (iA + 1, iB) not in visited:
                visited.add((iA + 1, iB))
                heapq.heappush(heap, (-(A[iA+1][0] + B[iB][0]), iA + 1, iB))
        if iB + 1 < M:
            if (iA, iB + 1) not in visited:
                visited.add((iA, iB + 1))
                heapq.heappush(heap, (-(A[iA][0] + B[iB+1][0]), iA, iB + 1))

# Call main() to execute solution
if __name__ == "__main__":
    main()