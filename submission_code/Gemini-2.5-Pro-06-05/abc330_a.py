# YOUR CODE HERE
def solve():
    """
    This function reads the input, solves the problem, and prints the output.
    """
    # Read the number of people (N) and the passing score (L) from the first line.
    # The input values are separated by a space, so we split the line and
    # map the resulting strings to integers.
    try:
        N, L = map(int, input().split())
    except (IOError, ValueError):
        # Handle potential empty lines or invalid input format, though
        # problem constraints suggest this won't happen.
        return

    # Read the list of scores (A) from the second line.
    # Similar to the first line, we split by space and map to integers.
    A = list(map(int, input().split()))

    # Initialize a counter to keep track of the number of people who passed.
    passed_count = 0

    # Iterate through each score in the list A.
    for score in A:
        # Check if the person's score is at least the passing score L.
        if score >= L:
            # If the score is sufficient, increment the counter.
            passed_count += 1

    # Print the final count of people who passed the exam.
    print(passed_count)

# It's good practice to wrap the main logic in a function and call it.
solve()