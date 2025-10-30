def main():
    import sys
    sys.setrecursionlimit(10**7)

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    P = list(map(lambda x: int(x) - 1, input_data[1:N+1]))  # zero-based
    A = list(map(lambda x: int(x), input_data[N+1:2*N+1]))  # keep values as 1-based but indices as 0-based

    # We will find the cycles in P. Each cycle c is a list of indices in that cycle.
    visited = [False]*N
    result = [0]*N  # to store final arrangement
    
    # Function to find the start index of the lexicographically smallest rotation using Booth's algorithm
    def booth_algorithm(arr):
        # Returns the starting index of the minimal lex rotation
        n = len(arr)
        f = [-1]*(2*n)  # Failure function
        s = 0  # least rotation of string found so far
        for i in range(1, 2*n):
            j = f[i - s - 1]
            while j != -1 and arr[(i % n)] != arr[((s + j + 1) % n)]:
                if arr[(i % n)] < arr[((s + j + 1) % n)]:
                    s = i - j - 1
                j = f[j]
            if arr[(i % n)] != arr[((s + j + 1) % n)]:
                if arr[(i % n)] < arr[((s + j + 1) % n)]:
                    s = i
                f[i - s] = -1
            else:
                f[i - s] = j + 1
        return s

    for i in range(N):
        if not visited[i]:
            cycle_indices = []
            current = i
            while not visited[current]:
                visited[current] = True
                cycle_indices.append(current)
                current = P[current]

            # Gather the A-values in this cycle
            cycle_vals = [A[idx] for idx in cycle_indices]
            # We'll find the minimal rotation of cycle_vals
            start = booth_algorithm(cycle_vals)
            L = len(cycle_indices)
            # This is the minimal-rotation array
            best_rotation = cycle_vals[start:] + cycle_vals[:start]

            # Place these in ascending order of cycle indices
            cycle_indices.sort()
            for j, idx in enumerate(cycle_indices):
                result[idx] = best_rotation[j]

    print(" ".join(map(str, result)))

# Call main
if __name__ == "__main__":
    main()