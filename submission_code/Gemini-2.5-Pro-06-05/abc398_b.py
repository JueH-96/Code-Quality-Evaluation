import collections

def main():
  """
  Reads 7 card values, determines if a full house can be formed, and prints the result.
  """
  try:
    # Read the 7 integers from a single line of input.
    cards = list(map(int, input().split()))

    # Use collections.Counter to count the occurrences of each card number.
    # For example, input '1 4 1 4 2 1 3' gives Counter({1: 3, 4: 2, 2: 1, 3: 1})
    counts = collections.Counter(cards)

    # We only care about the counts themselves, not the card numbers.
    # For the example above, the counts are [3, 2, 1, 1].
    frequencies = sorted(list(counts.values()))

    # A full house requires 3 cards of one kind and 2 of another.
    # This is only possible if the frequency distribution of the 7 cards
    # matches one of a few specific patterns. We can pre-determine these
    # patterns by analyzing the partitions of the number 7.
    # The sorted list of frequencies must be one of the following:
    # [2, 5]: Two of one kind, five of another. A full house can be formed.
    # [3, 4]: Three of one kind, four of another. A full house can be formed.
    # [1, 2, 4]: A single, a pair, and a four-of-a-kind. Full house possible.
    # [1, 3, 3]: A single and two different three-of-a-kinds. Full house possible.
    # [2, 2, 3]: Two different pairs and a three-of-a-kind. Full house possible.
    # [1, 1, 2, 3]: Two singles, a pair, and a three-of-a-kind. Full house possible.

    if (frequencies == [2, 5] or
        frequencies == [3, 4] or
        frequencies == [1, 2, 4] or
        frequencies == [1, 3, 3] or
        frequencies == [2, 2, 3] or
        frequencies == [1, 1, 2, 3]):
      print("Yes")
    else:
      print("No")

  except (IOError, ValueError) as e:
    # Handle potential input errors gracefully, though not expected for this problem's constraints.
    pass

if __name__ == "__main__":
  main()