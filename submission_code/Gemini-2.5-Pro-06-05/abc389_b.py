# YOUR CODE HERE
def solve():
  """
  Reads an integer X and finds the positive integer N such that N! = X.
  """
  # Read the input integer X from standard input.
  try:
    X = int(input())
  except (ValueError, EOFError):
    # Handle cases with invalid or no input, though not expected by problem spec.
    return

  # Initialize the variable to store the current factorial value.
  current_factorial = 1
  
  # The value of N will be small (at most 20), so we can iterate.
  # We loop from n = 1 up to a safe limit (e.g., 21).
  for n in range(1, 22):
    # In each step of the loop, we calculate n! based on (n-1)!.
    current_factorial *= n
    
    # Check if the calculated n! is equal to the input X.
    if current_factorial == X:
      # If they are equal, n is the answer we are looking for.
      print(n)
      # Since the problem guarantees a unique solution, we can stop here.
      break

solve()