def main():
    import sys
    from math import gcd

    n = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))
    A = list(map(int, sys.stdin.readline().split()))

    # Convert P to 0-based index
    P = [x - 1 for x in P]

    # Find cycles in permutation P
    visited = [False] * n
    cycles = []
    for i in range(n):
        if not visited[i]:
            current = i
            cycle = []
            while not visited[current]:
                visited[current] = True
                cycle.append(current)
                current = P[current]
            cycles.append(cycle)

    # Compute the order of the permutation (LCM of cycle lengths)
    order = 1
    for cycle in cycles:
        m = len(cycle)
        order = (order * m) // gcd(order, m)

    # Function to compute the array after k operations
    def apply_k_operations(k):
        res = A.copy()
        for i in range(n):
            # Apply P k times
            pos = i
            for _ in range(k):
                pos = P[pos]
            res[i] = A[pos]
        return res

    # Find the minimal array
    min_array = None
    for k in range(order):
        current = apply_k_operations(k)
        if min_array is None or current < min_array:
            min_array = current

    # Convert back to 1-based if necessary (no, the output is 1-based indices but the array is 0-based)
    # Wait, no. The problem statement says to output the array elements, not their indices.
    # So no need to adjust.

    # Output the minimal array
    print(' '.join(map(str, min_array)))

if __name__ == '__main__':
    main()