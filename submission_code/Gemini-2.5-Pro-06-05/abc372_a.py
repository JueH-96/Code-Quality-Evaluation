# YOUR CODE HERE
def solve():
  """
  Reads a string from standard input, removes all '.' characters,
  and prints the resulting string to standard output.
  """
  s = input()
  result = s.replace('.', '')
  print(result)

solve()