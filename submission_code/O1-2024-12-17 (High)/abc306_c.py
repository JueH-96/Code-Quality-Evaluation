def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # For each i in [1..N], store the positions (1-based) where i occurs.
    occurrences = [[] for _ in range(N+1)]
    for idx, val in enumerate(A):
        occurrences[val].append(idx+1)

    # f(i) is the middle position of i's three occurrences
    # occurrences[i] = [alpha, beta, gamma], so f(i) = beta = occurrences[i][1]
    mid_positions = [(occurrences[i][1], i) for i in range(1, N+1)]
    
    # Sort by the middle positions
    mid_positions.sort(key=lambda x: x[0])
    
    # Print the values in ascending order of their middle positions
    print(" ".join(str(x[1]) for x in mid_positions))

# Do not forget to call main()
main()