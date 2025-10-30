# YOUR CODE HERE
def solve():
  """
  Reads an integer N and a string S from standard input,
  then prints a new string where each character of S is duplicated.
  """
  # Read the integer N. It is not used in the logic but is part of the input format.
  N = int(input())
  
  # Read the string S.
  S = input()
  
  # Create a list or generator of doubled characters.
  # For each character `c` in the string `S`, we create a new string `c * 2`.
  # For example, if c is 'a', c * 2 is 'aa'.
  doubled_chars = (c * 2 for c in S)
  
  # Join the doubled characters together to form the final result string.
  # The "".join() method is an efficient way to concatenate a sequence of strings.
  result = "".join(doubled_chars)
  
  # Print the final string to standard output.
  print(result)

solve()