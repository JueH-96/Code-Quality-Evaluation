# YOUR CODE HERE
def solve():
  """
  Reads an integer N and prints the results of N penalty kicks.
  A kick succeeds ('o') unless its number is a multiple of 3, in which case it fails ('x').
  """
  try:
    # Read the number of penalty kicks from standard input.
    N = int(input())

    # Use a list comprehension to generate the sequence of results.
    # The loop runs from 1 to N (inclusive) to represent the kick numbers.
    # A conditional expression checks if the kick number 'i' is a multiple of 3.
    # 'x' for failure (multiple of 3), 'o' for success (otherwise).
    result_list = ['x' if i % 3 == 0 else 'o' for i in range(1, N + 1)]
    
    # Join the characters in the list to form the final output string.
    output_string = "".join(result_list)
    
    # Print the result to standard output.
    print(output_string)
    
  except (ValueError, EOFError):
    # Handle potential input errors, though not expected under problem constraints.
    pass

solve()