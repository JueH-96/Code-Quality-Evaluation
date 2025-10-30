import sys

def solve():
    # Read N and sequence A
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Read M and sequence B
    M = int(sys.stdin.readline())
    B = list(map(int, sys.stdin.readline().split()))

    # Read L and sequence C
    L = int(sys.stdin.readline())
    C = list(map(int, sys.stdin.readline().split()))

    # Read Q and sequence X
    Q = int(sys.stdin.readline())
    X = list(map(int, sys.stdin.readline().split()))

    # Precompute all possible sums a + b + c
    # Using a set for O(1) average time complexity for additions and lookups.
    # The maximum number of elements in the set is N*M*L = 100*100*100 = 1,000,000,
    # which is manageable.
    possible_sums = set()
    for a_val in A:
        for b_val in B:
            for c_val in C:
                possible_sums.add(a_val + b_val + c_val)

    # For each query X_i, check if it exists in the set of possible sums.
    # Store results in a list to print them efficiently at the end.
    results = []
    for x_val in X:
        if x_val in possible_sums:
            results.append("Yes")
        else:
            results.append("No")
    
    # Print all results at once, each on a new line.
    sys.stdout.write("
".join(results) + "
")

# Call the main solve function to execute the program.
solve()