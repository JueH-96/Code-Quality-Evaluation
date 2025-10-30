# YOUR CODE HERE
def solve():
  """
  Reads a string from standard input, separates its characters with spaces,
  and prints the result to standard output.
  """
  # Read the input string from standard input.
  s = input()
  
  # The `str.join()` method is a Pythonic way to join elements of an
  # iterable (like a string) with a specified separator.
  # Here, we use a space ' ' as the separator to join the characters of the string s.
  result = " ".join(s)
  
  # Print the final space-separated string to standard output.
  print(result)

# Call the function to run the solution.
solve()