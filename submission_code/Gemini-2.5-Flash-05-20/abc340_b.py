# YOUR CODE HERE
import sys

def solve():
    # Read the number of queries
    Q = int(sys.stdin.readline())

    # Initialize an empty list to represent sequence A
    A = []

    # Process each query
    for _ in range(Q):
        query_parts = list(map(int, sys.stdin.readline().split()))
        query_type = query_parts[0]

        if query_type == 1:
            # Type 1 query: Append x to the end of A
            x = query_parts[1]
            A.append(x)
        elif query_type == 2:
            # Type 2 query: Find the k-th value from the end of A
            # It's guaranteed that the length of A is at least k.
            k = query_parts[1]
            
            # Python's negative indexing allows direct access to the k-th element from the end.
            # A[-1] is the 1st from the end, A[-2] is the 2nd, and so on.
            # So, A[-k] directly gives the k-th element from the end.
            result = A[-k]
            sys.stdout.write(str(result) + '
')

# Call the solve function to run the program
solve()