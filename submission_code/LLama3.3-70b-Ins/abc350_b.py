def solve():
    # Read input from stdin
    N, Q = map(int, input().split())
    T = list(map(int, input().split()))

    # Initialize a set to keep track of teeth
    teeth = set(range(1, N + 1))

    # Perform treatments
    for t in T:
        if t in teeth:
            teeth.remove(t)
        else:
            teeth.add(t)

    # Print the number of teeth
    print(len(teeth))

# Call the solve function
solve()