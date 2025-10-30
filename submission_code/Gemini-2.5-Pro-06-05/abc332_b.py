def solve():
  """
  This function solves the problem by simulating the operations K times.
  """
  # Read the input values for K, G, and M from standard input.
  # K: the number of operations to perform.
  # G: the capacity of the glass.
  # M: the capacity of the mug.
  try:
    K, G, M = map(int, input().split())
  except (IOError, ValueError):
    # This handles potential input errors, though not strictly required by the problem format.
    return

  # Initialize the current amount of water in the glass and the mug.
  # Both start empty as per the problem description.
  glass = 0
  mug = 0

  # Perform the specified operation K times.
  for _ in range(K):
    # The conditions are checked in a specific order.

    # 1. If the glass is filled to its capacity G, discard all water from it.
    if glass == G:
      glass = 0
    
    # 2. Otherwise, if the mug is empty, fill the mug to its capacity M.
    elif mug == 0:
      mug = M
      
    # 3. Otherwise (glass is not full, mug is not empty), transfer water.
    else:
      # Calculate the amount to pour from the mug to the glass.
      # This is the minimum of the remaining space in the glass and the water available in the mug.
      pour_amount = min(G - glass, mug)
      
      # Update the water levels.
      glass += pour_amount
      mug -= pour_amount

  # After K operations, print the final amounts of water in the glass and mug,
  # separated by a space, to standard output.
  print(glass, mug)

# Execute the solution
solve()