def solve():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Compute B_i for i = 1 to N-1
    results = [A[i] * A[i+1] for i in range(N - 1)]

    # Print the results separated by spaces
    print(" ".join(map(str, results)))

# Call the solve() function
solve()