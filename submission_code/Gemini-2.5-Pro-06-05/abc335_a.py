# YOUR CODE HERE
def solve():
  """
  Reads a string S from standard input, changes its last character to '4',
  and prints the modified string.
  """
  # Read the input string from standard input.
  s = input()

  # Strings in Python are immutable. To modify the string, we need to create
  # a new one. We can do this by taking a slice of the original string
  # that includes all characters except the last one, and then concatenating
  # the new character '4' to it.

  # S[:-1] gets a substring from the beginning of S up to the character
  # just before the last one.
  prefix = s[:-1]

  # Concatenate the prefix with the new last character '4'.
  modified_s = prefix + '4'

  # Print the final modified string to standard output.
  print(modified_s)

solve()