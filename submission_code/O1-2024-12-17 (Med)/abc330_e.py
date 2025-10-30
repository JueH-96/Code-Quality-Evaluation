def main():
    import sys
    import heapq

    input_data = sys.stdin.read().strip().split()
    # Parse input
    N, Q = map(int, input_data[:2])
    A = list(map(int, input_data[2:2+N]))

    # Queries start after N elements of A
    queries = input_data[2+N:]
    # Each query consists of (i_k, x_k)
    # so total 2*Q elements
    # queries[0], queries[1] => i_1, x_1
    # queries[2], queries[3] => i_2, x_2
    # ...
    
    # Frequency array for values 0..N only
    freq = [0]*(N+1)
    
    # Build initial frequency
    for v in A:
        if v <= N:
            freq[v] += 1

    # We'll use a min-heap containing all candidates 0..N.
    # The MEX is the smallest i in this heap such that freq[i] == 0.
    h = list(range(N+1))
    heapq.heapify(h)

    # A helper function to get the current MEX (lazy removal).
    def get_mex():
        while freq[h[0]] != 0:
            heapq.heappop(h)
        return h[0]

    # Process queries
    idx = 0
    out = []
    for _ in range(Q):
        i_k = int(queries[idx]); x_k = int(queries[idx+1])
        idx += 2

        # Convert to 0-based index
        i_k -= 1

        # Decrease freq of old value if <= N
        old_val = A[i_k]
        if old_val <= N:
            freq[old_val] -= 1
            # If freq for old_val becomes 0, push to heap
            if freq[old_val] == 0:
                heapq.heappush(h, old_val)

        # Update A[i_k]
        A[i_k] = x_k

        # Increase freq of new value if <= N
        if x_k <= N:
            # If freq[x_k] was 0, it will now become 1,
            # so we do not add x_k to heap (since freq[x_k] != 0 anymore).
            freq[x_k] += 1

        # Determine current MEX
        mex = get_mex()
        out.append(str(mex))

    print("
".join(out))

# Don't forget to call main() at the end
if __name__ == "__main__":
    main()