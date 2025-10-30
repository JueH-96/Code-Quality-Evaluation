# YOUR CODE HERE
import sys
from collections import Counter

# Read input string from stdin
# .strip() removes any leading/trailing whitespace, including the newline character
S = sys.stdin.readline().strip()

# Use collections.Counter to count the frequency of each character in the string S.
# Example: if S = "banana", counts will be a Counter object like {'a': 3, 'n': 2, 'b': 1}
counts = Counter(S)

# Convert the Counter items (which are key-value pairs of character and frequency) into a list of tuples.
# Example: items might be [('b', 1), ('a', 3), ('n', 2)]
# The initial order of items in the list doesn't matter because we will sort it.
items = list(counts.items())

# Sort the list of (character, frequency) tuples.
# We use a custom key function `lambda item: (-item[1], item[0])` for sorting:
# 1. `-item[1]`: Sorts by frequency in descending order. We use the negative of the frequency (`item[1]`)
#    because `sorted()` sorts in ascending order by default. Sorting by negative frequency in ascending
#    order is equivalent to sorting by frequency in descending order. Characters with higher frequencies
#    will come earlier in the sorted list.
# 2. `item[0]`: Sorts by character in ascending (alphabetical) order. This is the secondary sort key.
#    If two characters have the same frequency (i.e., `-item[1]` is the same for both),
#    the one that comes earlier alphabetically (`item[0]`) will come first in the sorted list.
#
# Example walkthrough with S = "frequency":
# counts = {'f': 1, 'r': 1, 'e': 2, 'q': 1, 'u': 1, 'n': 1, 'c': 1, 'y': 1}
# items = [('f', 1), ('r', 1), ('e', 2), ('q', 1), ('u', 1), ('n', 1), ('c', 1), ('y', 1)]
# Keys generated: ('e', 2) -> (-2, 'e'), ('c', 1) -> (-1, 'c'), ('f', 1) -> (-1, 'f'), etc.
# Sorting these keys results in (-2, 'e') coming first, followed by keys like (-1, 'c'), (-1, 'f'), etc.
# The sorted list `sorted_items` will start with ('e', 2).
sorted_items = sorted(items, key=lambda item: (-item[1], item[0]))

# The first item in the `sorted_items` list `sorted_items[0]` is the tuple corresponding to the character that
# satisfies the condition: it has the highest frequency, and among those with the highest frequency,
# it is the earliest in alphabetical order.
# We need to print the character itself, which is the first element (index 0) of this tuple.
# Example: if sorted_items starts with ('e', 2), we print sorted_items[0][0] which is 'e'.
print(sorted_items[0][0])
# YOUR CODE HERE