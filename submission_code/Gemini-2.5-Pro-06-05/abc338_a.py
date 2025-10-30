# YOUR CODE HERE
def solve():
  """
  Reads a string from standard input and checks if it meets the specified capitalization criteria.
  """
  # Read the input string. The constraints guarantee it's non-empty.
  S = input()

  # The str.capitalize() method returns a version of the string with the first
  # character in uppercase and all other characters in lowercase.
  # This perfectly matches the condition we need to check.
  #
  # If the original string S is identical to its capitalized version,
  # it means S already meets the condition.
  if S == S.capitalize():
    print("Yes")
  else:
    print("No")

solve()