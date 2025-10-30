# YOUR CODE HERE
import sys

# Read the integer N from standard input
# N represents the length of the strings S and T.
n = int(sys.stdin.readline())

# Read the string S from standard input.
# .strip() removes any leading/trailing whitespace, including the newline character.
s = sys.stdin.readline().strip()

# Read the string T from standard input.
# .strip() removes any leading/trailing whitespace, including the newline character.
t = sys.stdin.readline().strip()

# Calculate the Hamming distance between S and T.
# The Hamming distance is the number of positions at which the corresponding characters are different.

# We can achieve this efficiently using the `zip` function and a generator expression within `sum()`.
# `zip(s, t)` creates an iterator that aggregates elements from S and T.
# It yields pairs of characters (s[0], t[0]), (s[1], t[1]), ..., (s[N-1], t[N-1]).
# The generator expression `(1 for char_s, char_t in zip(s, t) if char_s != char_t)`
# iterates through these pairs. For each pair where the characters `char_s` and `char_t`
# are different (`char_s != char_t`), it yields the value 1.
# The `sum()` function then calculates the total sum of these yielded 1s, which gives
# the total count of positions where the characters differ.
distance = sum(1 for char_s, char_t in zip(s, t) if char_s != char_t)

# Print the calculated Hamming distance to standard output.
print(distance)