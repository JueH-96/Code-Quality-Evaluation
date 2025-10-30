def main():
    import sys

    # Read all input
    data = sys.stdin.read().split()
    if not data:
        return

    # Parse the input values
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Dictionary to store the first occurrence of each color
    first_occurrence = {}
    count = 0

    # The list is 1-indexed according to the problem.
    for pos, color in enumerate(A, start=1):
        if color not in first_occurrence:
            # Record the position of the first occurrence of this color.
            first_occurrence[color] = pos
        else:
            # For second occurrence, check if there is exactly one person
            # between the two persons: i.e., second - first == 2.
            if pos - first_occurrence[color] == 2:
                count += 1

    # Output the count of colors that satisfy the condition.
    print(count)

# Call the main function.
if __name__ == '__main__':
    main()