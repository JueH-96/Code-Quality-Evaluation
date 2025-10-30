def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    visited = [False] * N
    swaps = []

    # Decompose the permutation into cycles and fix each cycle.
    for i in range(N):
        if not visited[i]:
            cycle = []
            x = i
            # Follow the cycle starting at x
            while not visited[x]:
                visited[x] = True
                cycle.append(x)
                x = A[x] - 1  # move to next index according to the permutation
            # Fix the cycle using (length_of_cycle - 1) swaps
            # by swapping the "pivot" cycle[0] with each other element in turn.
            L = len(cycle)
            if L > 1:
                for k in range(1, L):
                    swaps.append((cycle[0], cycle[k]))
                    A[cycle[0]], A[cycle[k]] = A[cycle[k]], A[cycle[0]]

    # Output the result
    print(len(swaps))
    for i, j in swaps:
        # Convert from 0-based index to 1-based for output
        print(i+1, j+1)

# Do not forget to call main() at the end
if __name__ == "__main__":
    main()