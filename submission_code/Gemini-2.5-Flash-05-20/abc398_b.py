import sys
from itertools import combinations
from collections import Counter

def solve():
    # Read the seven card values from standard input.
    # The input is a single line of space-separated integers.
    # Convert them to a list of integers.
    A = list(map(int, sys.stdin.readline().split()))

    # A full house consists of 5 cards: three cards of one rank (x)
    # and two cards of another rank (y), where x and y are different.

    # We need to determine if it's possible to choose *five* cards from the given seven
    # that form a full house.
    # The total number of ways to choose 5 cards from 7 is C(7, 5) = 21, which is a small number.
    # We can iterate through all possible combinations of 5 cards.

    # itertools.combinations(iterable, r) generates all unique combinations of
    # r elements from the iterable.
    for chosen_cards_tuple in combinations(A, 5):
        # For each combination of 5 cards:
        # Use collections.Counter to count the frequency of each card rank
        # within this specific set of 5 cards.
        counts = Counter(chosen_cards_tuple)

        # A full house must have exactly two distinct card ranks.
        # For example, if it's three '1's and two '4's, then len(counts) will be 2 (for 1 and 4).
        # If it's five '7's (e.g., 7 7 7 7 7), len(counts) would be 1, which is not a full house.
        if len(counts) == 2:
            # If there are two distinct ranks, get their counts.
            # For a full house, these counts must be 2 and 3 (in any order).
            # We sort the values to easily compare them, e.g., [3, 2] becomes [2, 3].
            count_values = sorted(counts.values())

            # Check if the sorted counts are exactly [2, 3].
            # This confirms that one rank appears 2 times and the other appears 3 times.
            if count_values == [2, 3]:
                # If these conditions are met, we have found a full house.
                # Print "Yes" and terminate the program, as we only need to find one possibility.
                print("Yes")
                return

    # If the loop finishes without finding any combination that forms a full house,
    # it means it's not possible.
    print("No")

# Call the solve function to run the program.
solve()