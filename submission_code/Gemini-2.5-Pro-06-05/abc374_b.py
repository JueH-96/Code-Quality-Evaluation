import sys
from itertools import zip_longest

def solve():
  """
  Reads two strings S and T and finds the first position where they differ.
  """
  # Read the two strings from standard input.
  # .strip() removes any trailing newline characters.
  S = sys.stdin.readline().strip()
  T = sys.stdin.readline().strip()

  # If the strings are identical, the answer is 0 as per the problem statement.
  if S == T:
    print(0)
    return

  # If the strings are not equal, we find the first differing position.
  # We use enumerate to get the 0-based index `i`.
  # We use zip_longest to iterate through both strings until the longest one is exhausted.
  # It pairs corresponding characters. If one string is shorter, it pairs the
  # remaining characters of the longer string with `None`.
  for i, (s_char, t_char) in enumerate(zip_longest(S, T)):
    # The first time the characters (or their existence) differ,
    # we have found the smallest position of difference.
    if s_char != t_char:
      # The problem asks for a 1-based position, so we print i + 1.
      print(i + 1)
      # We have found the answer, so we can stop the loop.
      break

solve()