def solve():
  """
  Solves the problem by checking if the sorted input string equals "ABC".
  """
  # Read the input string from standard input.
  s = input()

  # Sort the characters of the string. The sorted() function returns a list
  # of characters, which we join back into a string.
  sorted_s = "".join(sorted(s))

  # Check if the sorted string is equal to "ABC".
  if sorted_s == "ABC":
    print("Yes")
  else:
    print("No")

# Execute the solution function.
solve()