def insert_numbers(sequence):
    """
    Inserts numbers into the sequence according to the given procedure.

    Args:
    sequence (list): The input sequence of integers.

    Returns:
    list: The modified sequence after the procedure ends.
    """
    while True:
        # Check if every pair of adjacent terms has an absolute difference of 1
        if all(abs(sequence[i] - sequence[i + 1]) == 1 for i in range(len(sequence) - 1)):
            break

        # Find the pair of adjacent terms nearest to the beginning of the sequence
        # whose absolute difference is not 1
        for i in range(len(sequence) - 1):
            if abs(sequence[i] - sequence[i + 1]) != 1:
                # Insert numbers between the pair of adjacent terms
                if sequence[i] < sequence[i + 1]:
                    sequence = sequence[:i + 1] + list(range(sequence[i] + 1, sequence[i + 1])) + sequence[i + 1:]
                else:
                    sequence = sequence[:i + 1] + list(range(sequence[i] - 1, sequence[i + 1], -1)) + sequence[i + 1:]
                break

    return sequence


def main():
    # Read the input from stdin
    N = int(input())
    A = list(map(int, input().split()))

    # Insert numbers into the sequence
    result = insert_numbers(A)

    # Print the result to stdout
    print(*result)


if __name__ == "__main__":
    main()