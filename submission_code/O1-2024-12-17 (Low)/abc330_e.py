def main():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    N, Q = map(int, input_data[:2])
    A = list(map(int, input_data[2:2+N]))
    queries = input_data[2+N:]
    
    # We only care about counts of integers in [0..N+1], because mex cannot exceed N+1.
    counts = [0]*(N+2)
    
    # Build initial counts
    for val in A:
        if val <= N+1:
            counts[val] += 1
    
    # A min-heap that will store all candidates for mex (those whose counts can be 0).
    # We'll keep [0..N+1] in the candidate set initially, because any of these could become the mex.
    candidates = []
    for i in range(N+2):
        if counts[i] == 0:
            heapq.heappush(candidates, i)

    def get_mex():
        # Pop until we find a value i that still has counts[i]==0.
        # Then push it back (so that in the future if it remains 0, we can find it again).
        while True:
            i = heapq.heappop(candidates)
            if counts[i] == 0:
                heapq.heappush(candidates, i)
                return i
            # else, keep popping

    # Process queries
    idx = 0
    out = []
    for _ in range(Q):
        i_k = int(queries[idx]) - 1  # convert to 0-based
        x_k = int(queries[idx+1])
        idx += 2
        
        # Decrease count of old value if relevant
        old_val = A[i_k]
        if old_val <= N+1:
            counts[old_val] -= 1
            if counts[old_val] == 0:
                heapq.heappush(candidates, old_val)
        
        # Increase count of new value if relevant
        if x_k <= N+1:
            if counts[x_k] == 0:
                heapq.heappush(candidates, x_k)
            counts[x_k] += 1
        
        # Update A
        A[i_k] = x_k
        
        # Print the mex
        out.append(str(get_mex()))
    
    print("
".join(out))

# Don't forget to call main()
if __name__ == "__main__":
    main()