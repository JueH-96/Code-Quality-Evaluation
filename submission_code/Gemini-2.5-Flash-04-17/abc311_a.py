# Read the integer N, which is the length of the string S.
N = int(input())

# Read the string S from standard input.
S = input()

# Initialize a set to keep track of the unique characters encountered
# as we scan the string from left to right.
# We are specifically looking for the appearance of 'A', 'B', and 'C'.
seen_chars = set()

# Iterate through the string S using its index 'i'.
# The index 'i' goes from 0 up to N-1.
# The number of characters examined up to the current index i (inclusive) is i + 1.
for i in range(N):
    # Add the character at the current index S[i] to the set of seen characters.
    # Sets only store unique elements, so duplicate characters don't change the set after the first appearance.
    seen_chars.add(S[i])

    # Check if the set of seen characters now contains all three target characters: 'A', 'B', and 'C'.
    # Since the problem guarantees that S consists only of 'A', 'B', and 'C',
    # having seen all three is equivalent to the set being exactly {'A', 'B', 'C'}.
    if seen_chars == {'A', 'B', 'C'}:
        # If the set contains all three characters, it means we have found the point
        # where all required characters have appeared at least once.
        # Because we are iterating from left to right, this is the first time this condition is met.
        # The number of characters checked to reach this point is i + 1.
        print(i + 1)
        # We have found the minimum length, so we can stop the loop.
        break