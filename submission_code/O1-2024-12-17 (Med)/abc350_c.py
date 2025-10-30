def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A_list = list(map(int, input_data[1:]))

    # We'll convert A_list to 1-based indexing for convenience:
    # A[i] will store the element at position i, for i=1..N
    A = [0] + A_list

    visited = [False] * (N + 1)
    swaps = []

    # Find cycles in the permutation i -> A[i], then fix each cycle
    for i in range(1, N + 1):
        if not visited[i]:
            cycle = []
            c = i
            while not visited[c]:
                visited[c] = True
                cycle.append(c)
                c = A[c]

            # Fix the cycle with (L-1) swaps if cycle length > 1
            L = len(cycle)
            if L > 1:
                # A standard way: swap(A[cycle[0]], A[cycle[k]]) for k=1..L-1
                for k in range(1, L):
                    i1 = cycle[0]
                    i2 = cycle[k]
                    swaps.append((i1, i2))
                    # Perform the swap in A
                    A[i1], A[i2] = A[i2], A[i1]

    # Output the result
    print(len(swaps))
    for i1, i2 in swaps:
        print(i1, i2)

# Do not forget to call main()!
if __name__ == "__main__":
    main()