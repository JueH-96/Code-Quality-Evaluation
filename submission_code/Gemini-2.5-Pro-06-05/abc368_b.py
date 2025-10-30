import sys

def main():
    """
    Reads input from stdin, simulates the specified process,
    and prints the total number of operations to stdout.
    """
    # Read the number of elements, N. It's specified in the input format
    # but not strictly necessary for the logic, as len(a) could be used.
    try:
        n_str = sys.stdin.readline()
        if not n_str: return
        _ = int(n_str)
    except (ValueError, IndexError):
        # Gracefully exit on malformed or empty input for N.
        return

    # Read the sequence of N integers into a list.
    try:
        a_str = sys.stdin.readline()
        if not a_str: return
        a = list(map(int, a_str.split()))
    except (ValueError, IndexError):
        # Gracefully exit on malformed or empty input for the list A.
        return

    # Initialize a counter for the operations.
    operations = 0

    # The simulation loop continues as long as there are more than one
    # positive element in the list 'a'.
    while sum(1 for x in a if x > 0) > 1:
        # Each iteration of the loop represents one operation.
        operations += 1

        # Step 1: Sort the list 'a' in descending order as required.
        a.sort(reverse=True)

        # Step 2: Decrease the two largest elements (now at indices 0 and 1) by 1.
        # The problem constraints (N >= 2) guarantee that a[0] and a[1] are always accessible.
        a[0] -= 1
        a[1] -= 1

    # After the loop terminates, print the final count of operations.
    print(operations)

if __name__ == "__main__":
    main()