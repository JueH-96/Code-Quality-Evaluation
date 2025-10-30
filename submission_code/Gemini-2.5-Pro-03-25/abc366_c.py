# YOUR CODE HERE
import sys

# Function to handle the core logic
def solve():
    # Read the number of queries
    q = int(sys.stdin.readline())

    # Use a dictionary to store the counts of each integer (ball value) in the bag.
    # Key: integer written on the ball
    # Value: count of balls with that integer
    # This dictionary will only store keys for integers that are currently present
    # in the bag (i.e., count > 0).
    counts = {}

    # Process each query one by one in the given order
    for _ in range(q):
        # Read the query line and parse the components (query type and value x if applicable)
        # Using map(int, ...) is a concise way to convert space-separated strings to integers
        query = list(map(int, sys.stdin.readline().split()))
        query_type = query[0]

        if query_type == 1:
            # Type 1 query: Put one ball with integer x into the bag
            x = query[1]
            # Increment the count of integer x in the dictionary.
            # `counts.get(x, 0)` returns the current count of x if it exists,
            # or 0 if x is not yet in the dictionary (i.e., first ball of this value).
            # Then, we add 1 to this count and update the dictionary.
            counts[x] = counts.get(x, 0) + 1
        elif query_type == 2:
            # Type 2 query: Remove one ball with integer x from the bag
            x = query[1]
            # The problem guarantees that a ball with integer x exists in the bag.
            # Therefore, x must be a key in the `counts` dictionary, and counts[x] >= 1.
            # Decrement the count of integer x.
            counts[x] -= 1
            # If the count of x becomes 0 after decrementing, it means there are
            # no more balls with integer x left in the bag.
            # We remove the key x from the dictionary to accurately reflect the
            # set of distinct integers present.
            if counts[x] == 0:
                del counts[x]
        elif query_type == 3:
            # Type 3 query: Print the number of different integers written on the balls in the bag.
            # Since the `counts` dictionary only stores keys for integers currently
            # present in the bag (count > 0), the number of distinct integers
            # is simply the number of keys in the dictionary.
            # `len(counts)` gives the size of the dictionary (number of key-value pairs).
            print(len(counts))

# Call the solve function to execute the logic when the script runs
solve()