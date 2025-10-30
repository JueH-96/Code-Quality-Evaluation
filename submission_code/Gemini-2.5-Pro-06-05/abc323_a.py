# YOUR CODE HERE
import sys

def solve():
  """
  Reads a string of length 16 and checks a condition on its characters.
  """
  # Read the input string from standard input.
  S = sys.stdin.readline().strip()

  # The problem asks to check if the i-th character is '0' for all even i from 2 to 16.
  # This uses 1-based indexing. In Python's 0-based indexing, these are indices
  # 1, 3, 5, 7, 9, 11, 13, and 15.

  # We can use Python's string slicing to extract all characters at these positions at once.
  # S[1::2] starts at index 1 and takes every second character until the end.
  relevant_chars = S[1::2]

  # The condition is met if all these characters are '0'. This is equivalent to
  # the character '1' not being present in the extracted slice, given the
  # constraint that the string only contains '0's and '1's.
  if '1' in relevant_chars:
    print("No")
  else:
    print("Yes")

solve()