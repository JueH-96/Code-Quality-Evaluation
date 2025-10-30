# YOUR CODE HERE
def solve():
  """
  Reads an integer N from stdin, concatenates N copies of the digit N,
  and prints the resulting string to stdout.
  """
  # Read the integer N from standard input
  N = int(input())
  
  # Convert the integer N to its string representation
  digit_n_as_string = str(N)
  
  # Use string multiplication to repeat the digit string N times
  result = digit_n_as_string * N
  
  # Print the final result to standard output
  print(result)

solve()