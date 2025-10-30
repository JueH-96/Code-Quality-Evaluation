def count_unique_sticks(n, sticks):
    """
    Counts the number of unique sticks.

    Args:
    n (int): The number of sticks.
    sticks (list): A list of strings representing the sticks.

    Returns:
    int: The number of unique sticks.
    """
    unique_sticks = set()
    for stick in sticks:
        # Normalize the stick by taking the lexicographically smaller string between the stick and its reversal
        normalized_stick = min(stick, stick[::-1])
        unique_sticks.add(normalized_stick)
    return len(unique_sticks)


def main():
    # Read the number of sticks
    n = int(input())

    # Read the sticks
    sticks = [input() for _ in range(n)]

    # Count the unique sticks
    unique_stick_count = count_unique_sticks(n, sticks)

    # Print the result
    print(unique_stick_count)


if __name__ == "__main__":
    main()