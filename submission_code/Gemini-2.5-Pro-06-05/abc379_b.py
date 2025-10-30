# YOUR CODE HERE
def solve():
    """
    Reads input, solves the problem, and prints the output.
    """
    try:
        # Read N and K from the first line of standard input.
        # N is the total number of teeth, K is the number of consecutive healthy
        # teeth needed to eat one strawberry.
        N, K = map(int, input().split())

        # Read the string S representing the condition of the teeth.
        S = input()
    except (IOError, ValueError):
        # Handle potential I/O errors or empty input.
        return

    # The 'X's (cavities) act as separators. We can solve the problem for each
    # contiguous block of 'O's (healthy teeth) independently.
    # We split the string by 'X' to get a list of these blocks.
    healthy_blocks = S.split('X')

    # Initialize a counter for the total number of strawberries.
    total_strawberries = 0

    # Iterate through each block of healthy teeth.
    for block in healthy_blocks:
        # For a block of healthy teeth of length `L`, the maximum number of
        # non-overlapping groups of size K is `L // K`. Each group
        # allows eating one strawberry.
        length_of_block = len(block)
        
        # Add the number of strawberries from the current block to the total.
        # Integer division `//` correctly handles cases where length < K (result is 0).
        total_strawberries += length_of_block // K

    # Print the final result to standard output.
    print(total_strawberries)

solve()