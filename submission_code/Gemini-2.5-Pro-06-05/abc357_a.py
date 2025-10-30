# YOUR CODE HERE
def solve():
  """
  Reads the problem input, simulates the disinfection process,
  and prints the number of aliens who successfully disinfect all their hands.
  """
  
  # Read the number of aliens (N) and the initial disinfectant capacity (M).
  # The 'n' variable is not strictly needed for the logic but is part of the input spec.
  try:
    n, m = map(int, input().split())
  except (IOError, ValueError):
    # Exit gracefully on malformed input.
    return

  # Read the list of hand counts for each alien.
  try:
    hands_list = list(map(int, input().split()))
  except (IOError, ValueError):
    # Exit gracefully on malformed input.
    return

  # Initialize a counter for successful aliens.
  successful_count = 0

  # Iterate through each alien in the order they arrive.
  for hands in hands_list:
    # Check if there is enough disinfectant for the current alien.
    if m >= hands:
      # If yes, subtract the amount used and count the alien as successful.
      m -= hands
      successful_count += 1
    else:
      # If not, this alien fails. No subsequent alien can succeed,
      # so we can stop processing.
      break
  
  # Print the final count to standard output.
  print(successful_count)

# Execute the solution.
solve()