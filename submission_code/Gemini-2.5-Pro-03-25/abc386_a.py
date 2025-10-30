# YOUR CODE HERE
import sys
from collections import Counter

def solve():
    """
    Reads four integers representing card values from standard input.
    Determines if adding one more card can result in a Full House.
    A Full House consists of five cards: three cards of one value (x)
    and two cards of another value (y), where x != y.
    Prints "Yes" if a Full House can be formed, "No" otherwise.
    """
    # Read the four integers from standard input
    # The input is expected to be in the format: A B C D
    nums = list(map(int, sys.stdin.readline().split()))

    # Count the frequency of each number using collections.Counter
    # This creates a dictionary-like object where keys are the distinct numbers
    # and values are their counts in the input list `nums`.
    # Example: If nums = [7, 7, 7, 1], counts will be Counter({7: 3, 1: 1}).
    counts = Counter(nums)

    # Get the frequency counts as a list.
    # This list contains the counts of each unique number found in `nums`.
    # Example: For Counter({7: 3, 1: 1}), counts.values() gives dict_values([3, 1]).
    # We convert it to a list: [3, 1].
    # Example: For Counter({3: 2, 5: 2}), list(counts.values()) gives [2, 2].
    count_values = list(counts.values())

    # Sort the list of frequency counts. This step is crucial to ensure
    # that the order of counts doesn't affect the comparison.
    # For example, [3, 1] is sorted to [1, 3]. [2, 2] remains [2, 2].
    # This provides a canonical representation for the distribution of counts.
    count_values.sort()

    # Check the condition for forming a Full House by adding one card.
    # A Full House requires 5 cards with counts {3, 2} for two distinct numbers.
    # We start with 4 cards. We need to determine if adding one card can
    # result in the {3, 2} count distribution.
    # This is only possible if the initial 4 cards have counts that are
    # "one step away" from {3, 2}.
    #
    # Let's analyze the possible counts distributions for the initial 4 cards:
    # 1. Counts {4} (e.g., 8 8 8 8): Sorted counts [4]. Adding any card results in {5} or {4, 1}. Neither is a Full House.
    # 2. Counts {3, 1} (e.g., 7 7 7 1): Sorted counts [1, 3]. Adding the card with count 1 results in {3, 2}. This is a Full House.
    # 3. Counts {2, 2} (e.g., 3 3 5 5): Sorted counts [2, 2]. Adding either existing card results in {3, 2}. This is a Full House.
    # 4. Counts {2, 1, 1} (e.g., 1 1 2 3): Sorted counts [1, 1, 2]. Adding any card results in {3, 1, 1}, {2, 2, 1}, or {2, 1, 1, 1}. None is a Full House.
    # 5. Counts {1, 1, 1, 1} (e.g., 1 2 3 4): Sorted counts [1, 1, 1, 1]. Adding any card results in {2, 1, 1, 1} or {1, 1, 1, 1, 1}. None is a Full House.
    #
    # Therefore, a Full House can be formed if and only if the sorted counts
    # of the initial 4 cards are exactly [1, 3] or [2, 2].

    if count_values == [1, 3] or count_values == [2, 2]:
        # If the sorted counts match either of the required patterns, print "Yes".
        print("Yes")
    else:
        # Otherwise, it's impossible to form a Full House by adding one card, so print "No".
        print("No")

# Call the solve function to execute the main logic of the program.
solve()