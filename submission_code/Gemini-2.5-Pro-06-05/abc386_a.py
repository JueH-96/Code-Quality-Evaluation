import collections
import sys

def solve():
    """
    Reads four card numbers and determines if a Full House can be formed by adding one more card.
    """
    # Read a single line of space-separated integers from standard input.
    try:
        cards = list(map(int, sys.stdin.readline().split()))
    except (ValueError, IndexError):
        # This handles cases of empty or malformed input, though not expected by the problem constraints.
        return

    # Use collections.Counter to get the frequency of each card value.
    # e.g., for [7, 7, 7, 1], counts will be Counter({7: 3, 1: 1})
    counts = collections.Counter(cards)

    # Get the list of frequencies. For the example above, this would be [3, 1].
    frequencies = list(counts.values())

    # Sort the frequencies for a consistent representation, e.g., [1, 3].
    frequencies.sort()

    # A full house can be formed if and only if the initial four cards consist of:
    # 1. Three of a kind and a single card (frequencies [1, 3]).
    # 2. Two pairs (frequencies [2, 2]).
    # In any other case, it's not possible.
    if frequencies == [1, 3] or frequencies == [2, 2]:
        print("Yes")
    else:
        print("No")

solve()