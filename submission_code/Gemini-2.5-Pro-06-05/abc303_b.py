import sys

def solve():
    """
    This function solves the "Bad Mood" problem by counting pairs that never stood together.
    It reads from stdin and prints the result to stdout.
    """

    # Read the number of people (N) and the number of photos (M) from the first line.
    try:
        line = sys.stdin.readline()
        if not line: return # Handle empty input
        N, M = map(int, line.split())
    except (ValueError, IndexError):
        return

    # A set is used to store all unique pairs of people who have stood next to each other.
    # Using a set is efficient and automatically handles duplicates.
    adjacent_pairs = set()

    # Process each of the M photos.
    for _ in range(M):
        # Read the lineup for the current photo.
        lineup = list(map(int, sys.stdin.readline().split()))

        # A Pythonic way to iterate over adjacent elements in a list is to use zip.
        # zip(lineup, lineup[1:]) pairs up each element with its successor.
        for p1, p2 in zip(lineup, lineup[1:]):
            # To treat (x, y) and (y, x) as the same pair, we create a canonical representation
            # by sorting the elements. A tuple is used because it's hashable and can be
            # added to a set.
            pair = tuple(sorted((p1, p2)))
            adjacent_pairs.add(pair)

    # The total number of possible distinct pairs from N people is "N choose 2".
    total_possible_pairs = N * (N - 1) // 2

    # The number of pairs that may be in a bad mood are those who never stood together.
    # This is the total count of pairs minus the count of pairs that were adjacent.
    bad_mood_count = total_possible_pairs - len(adjacent_pairs)

    # Print the final answer.
    print(bad_mood_count)

solve()