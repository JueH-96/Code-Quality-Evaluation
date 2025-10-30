def main():
    import sys
    import heapq
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])
    L = int(data[2])
    
    # Read the prices of main dishes and side dishes.
    a = list(map(int, data[3:3+N]))
    b = list(map(int, data[3+N:3+N+M]))
    
    # Sort main dishes in descending order with tracking of original indices.
    Adesc = sorted([(a[i], i) for i in range(N)], key=lambda x: x[0], reverse=True)
    # Sort side dishes in descending order with tracking of original indices.
    Bdesc = sorted([(b[j], j) for j in range(M)], key=lambda x: x[0], reverse=True)
    
    # Create rank arrays mapping original indices to their positions in the sorted arrays.
    rankA = [0] * N
    for i, (_, orig_i) in enumerate(Adesc):
        rankA[orig_i] = i
    
    rankB = [0] * M
    for j, (_, orig_j) in enumerate(Bdesc):
        rankB[orig_j] = j
    
    # Initialize data structure to hold the disallowed combinations:
    # disallowed[i] will be the set of side-dish sorted indices that don't go well with main dish i.
    disallowed = [set() for _ in range(N)]
    
    # Parse the L disallowed (c, d) pairs.
    idx = 3 + N + M
    for _ in range(L):
        c = int(data[idx]); d = int(data[idx+1])
        idx += 2
        c -= 1  # convert to 0-based
        d -= 1
        # Map to sorted indices
        i0 = rankA[c]
        j0 = rankB[d]
        disallowed[i0].add(j0)
    
    # We will maintain a pointer array where pointer[i] indicates
    # which side dish index (in sorted order) we are currently trying for main dish i.
    pointer = [0]*N
    
    # Build a max-heap by storing negative sums (since Python has only a min-heap).
    # We'll put (-(Adesc[i] + Bdesc[0]), i) for all i.
    heap = []
    for i in range(N):
        if M > 0:
            sum_val = Adesc[i][0] + Bdesc[0][0]
            heapq.heappush(heap, (-sum_val, i))
    
    # Repeatedly pop the largest sum and check if that pair is disallowed.
    while heap:
        neg_sum, i = heapq.heappop(heap)
        j = pointer[i]
        # If we've exhausted side dishes for this main dish, just continue
        # (though by problem constraints, we should eventually find a valid one).
        if j >= M:
            continue
        
        if j not in disallowed[i]:
            # Found a valid (not disallowed) pair
            print(-neg_sum)
            return
        else:
            # The pair is disallowed, so move to the next side dish for main i.
            pointer[i] += 1
            j_next = pointer[i]
            if j_next < M:
                sum_val = Adesc[i][0] + Bdesc[j_next][0]
                heapq.heappush(heap, (-sum_val, i))
    
    # Given problem constraints, we should always find an answer before the heap is empty.
    # This line is a fallback if something unexpected happens.
    print(0)

# Call main() at the end.
if __name__ == "__main__":
    main()