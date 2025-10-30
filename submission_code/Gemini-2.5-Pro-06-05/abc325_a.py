# YOUR CODE HERE
import sys

def solve():
  """
  Reads a surname and a first name, then prints the surname with the "san" honorific.
  """
  # Read the space-separated input strings S and T from a single line.
  s, t = sys.stdin.readline().split()

  # The problem asks to print the surname (S), a space, and "san".
  # The first name (T) is given but not used in the output.
  # We use an f-string for clean and efficient string formatting.
  print(f"{s} san")

solve()