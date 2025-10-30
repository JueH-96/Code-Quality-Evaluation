def solve():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Positions array to quickly find where each value (1..N) currently is.
    # pos[x] = index of value x in A.
    # We'll keep everything 0-based internally, but remember to output 1-based indices.
    pos = [0]*(N+1)
    for i in range(N):
        pos[A[i]] = i

    operations = []
    # We'll fix A[0], A[1], ..., A[N-2] in order. The last one must then be in place as it's a permutation.
    for i in range(N-1):
        correct_value = i+1           # the value that should be at index i
        if A[i] != correct_value:
            # Find where 'correct_value' is
            p = pos[correct_value]
            # Swap in A
            A[i], A[p] = A[p], A[i]
            # Update positions
            pos[A[i]] = i
            pos[A[p]] = p
            # Record the operation (convert to 1-based indexing)
            operations.append((i+1, p+1))

    # Output the result
    print(len(operations))
    for op in operations:
        print(op[0], op[1])

# Let's call solve() to handle the I/O.
if __name__ == "__main__":
    solve()