def solve():
    """
    Reads the input, finds the missing integers, and prints the output as specified.
    """
    # Read N and M from the first line of standard input.
    # N is the upper bound of the range (1 to N).
    # M is the number of integers in the given sequence A.
    try:
        N, M = map(int, input().split())
    except (ValueError, EOFError):
        # Handle cases with malformed or empty input.
        return

    # Read the sequence A from the second line.
    # The constraint 1 <= M <= N guarantees this line is not empty.
    # We store the numbers in a set for efficient membership testing.
    try:
        given_numbers = set(map(int, input().split()))
    except (ValueError, EOFError):
        # If the second line is malformed or missing (contrary to constraints),
        # treat the given sequence as empty.
        given_numbers = set()

    # Use a list comprehension to generate a list of integers from 1 to N
    # that are not present in the `given_numbers` set.
    # The iteration from 1 to N ensures the resulting list is sorted in ascending order.
    missing_numbers = [i for i in range(1, N + 1) if i not in given_numbers]

    # Print the count of missing numbers.
    print(len(missing_numbers))

    # If the list of missing numbers is not empty, print its elements on a new line,
    # separated by spaces. The `*` operator unpacks the list.
    # This handles the requirement to leave the second line empty if no numbers are missing.
    if missing_numbers:
        print(*missing_numbers)

solve()