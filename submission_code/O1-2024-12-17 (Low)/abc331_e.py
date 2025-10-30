def main():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    N, M, L = map(int, input_data[:3])
    pt = 3
    A = list(map(int, input_data[pt:pt+N]))
    pt += N
    B = list(map(int, input_data[pt:pt+M]))
    pt += M
    
    forbidden = set()
    for i in range(L):
        ci = int(input_data[pt]) - 1  # make zero-based
        di = int(input_data[pt+1]) - 1
        pt += 2
        forbidden.add((ci, di))
    
    # Sort main dishes and side dishes in descending order, keeping original indices
    # A_sorted[i] = (price, original_index)
    A_sorted = sorted([(val, i) for i, val in enumerate(A)], key=lambda x: x[0], reverse=True)
    B_sorted = sorted([(val, j) for j, val in enumerate(B)], key=lambda x: x[0], reverse=True)
    
    # Max-heap approach (Python's heapq is a min-heap, so we store negative sums)
    # We'll push tuples (-sum, iA, iB), where iA, iB are indices in the sorted arrays
    visited = set()
    heap = []
    
    # Push the top candidate
    top_sum = A_sorted[0][0] + B_sorted[0][0]
    heapq.heappush(heap, (-top_sum, 0, 0))
    visited.add((0, 0))
    
    while heap:
        neg_sum, iA, iB = heapq.heappop(heap)
        s = -neg_sum
        origA = A_sorted[iA][1]
        origB = B_sorted[iB][1]
        
        # If this pair is not forbidden, we have our answer
        if (origA, origB) not in forbidden:
            print(s)
            return
        
        # Otherwise, expand neighbors
        # Option 1: move to next main dish
        if iA + 1 < N:
            nxt = (iA + 1, iB)
            if nxt not in visited:
                visited.add(nxt)
                sum_next = A_sorted[iA+1][0] + B_sorted[iB][0]
                heapq.heappush(heap, (-sum_next, iA+1, iB))
        
        # Option 2: move to next side dish
        if iB + 1 < M:
            nxt = (iA, iB + 1)
            if nxt not in visited:
                visited.add(nxt)
                sum_next = A_sorted[iA][0] + B_sorted[iB+1][0]
                heapq.heappush(heap, (-sum_next, iA, iB+1))

# Call main() to ensure it runs.
if __name__ == "__main__":
    main()