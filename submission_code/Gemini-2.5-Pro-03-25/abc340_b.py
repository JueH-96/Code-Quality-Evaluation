# YOUR CODE HERE
import sys

def solve():
    # Read the number of queries
    q = int(sys.stdin.readline())

    # Initialize an empty list to represent the sequence A
    a = []

    # Process each query
    for _ in range(q):
        # Read the query line and split it into parts
        # Convert parts to integers using map
        query = list(map(int, sys.stdin.readline().split()))

        # The first part is the query type
        query_type = query[0]

        if query_type == 1:
            # Type 1 query: Append x to the end of A
            # The second part is the value x
            x = query[1]
            a.append(x)
        elif query_type == 2:
            # Type 2 query: Find the k-th value from the end of A
            # The second part is the value k
            k = query[1]

            # Python lists support negative indexing.
            # a[-1] refers to the last element (1st from the end).
            # a[-2] refers to the second-to-last element (2nd from the end).
            # ...
            # a[-k] refers to the k-th element from the end.
            # The problem guarantees that the length of A is at least k,
            # so a[-k] is always a valid index.
            print(a[-k])

# Call the solve function to run the logic
solve()