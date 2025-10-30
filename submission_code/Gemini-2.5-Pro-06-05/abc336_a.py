# YOUR CODE HERE
def solve():
  """
  Reads an integer N and prints the Dragon String of level N.
  """
  # Read the integer N from standard input.
  # The input is guaranteed to be a single integer.
  try:
    N = int(input())
  except (ValueError, EOFError):
    # This handles cases where input is not a valid integer or is empty.
    # Given the problem constraints, this is unlikely to occur in testing.
    return

  # A Dragon String of level N is defined as:
  # 1. One uppercase 'L'
  # 2. N occurrences of lowercase 'o'
  # 3. One lowercase 'n'
  # 4. One lowercase 'g'
  #
  # We can construct this string using concatenation.
  # The repeating 'o' part can be generated using string multiplication.
  
  o_part = 'o' * N
  dragon_string = 'L' + o_part + 'ng'

  # Print the final string to standard output.
  print(dragon_string)

# Execute the solution
solve()