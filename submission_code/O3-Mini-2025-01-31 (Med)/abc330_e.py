def main():
    import sys, heapq
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    q = int(next(it))
    # We'll only need to track numbers in the range 0 to n+1 because mex(A) is always ≤ n+1.
    limit = n + 2
    A = [0] * n
    freq = [0] * limit

    # Read the initial sequence and update frequency for numbers that fall in [0, limit)
    for i in range(n):
        val = int(next(it))
        A[i] = val
        if 0 <= val < limit:
            freq[val] += 1

    # Build a min-heap of missing numbers: i is missing if freq[i] == 0.
    heap = []
    for i in range(limit):
        if freq[i] == 0:
            heapq.heappush(heap, i)

    # Process each query
    output = []
    for _ in range(q):
        idx = int(next(it)) - 1  # Convert 1-indexed to 0-indexed.
        new_val = int(next(it))
        old_val = A[idx]
        A[idx] = new_val

        # Update frequency for the old value if it is in range
        if 0 <= old_val < limit:
            freq[old_val] -= 1
            if freq[old_val] == 0:
                heapq.heappush(heap, old_val)

        # Update frequency for the new value if it is in range.
        if 0 <= new_val < limit:
            freq[new_val] += 1

        # The answer (mex) is the smallest number with frequency 0.
        # Use lazy deletion on the heap: pop values that are no longer missing.
        while heap and freq[heap[0]] != 0:
            heapq.heappop(heap)
        output.append(str(heap[0]))

    sys.stdout.write("
".join(output))

if __name__ == '__main__':
    main()