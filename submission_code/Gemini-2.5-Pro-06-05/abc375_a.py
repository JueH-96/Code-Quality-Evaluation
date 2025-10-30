# YOUR CODE HERE
import sys

def solve():
  """
  Reads the input, solves the problem, and prints the answer.
  """
  try:
    # Read the number of seats, N.
    # Using sys.stdin.readline() for potentially faster I/O.
    N = int(sys.stdin.readline())
    
    # Read the state of the seats as a string, S.
    # .strip() removes leading/trailing whitespace, including the newline character.
    S = sys.stdin.readline().strip()
  except (IOError, ValueError):
    # Handle cases with empty input or malformed input gracefully.
    # For this problem, constraints guarantee valid input, but this is good practice.
    print(0)
    return

  # Initialize a counter for the occurrences of the pattern.
  count = 0
  
  # The pattern "#.#" has a length of 3.
  # We need to iterate through the string to check for this pattern.
  # The loop for the starting index `i` will go from 0 up to N-3.
  # `range(N - 2)` generates integers from 0, 1, ..., N-3.
  # If N < 3, range(N-2) will be empty, and the loop won't run,
  # correctly resulting in a count of 0.
  for i in range(N - 2):
    # We are looking for the pattern where:
    # - Seat at index `i` is occupied ('#')
    # - Seat at index `i+1` is unoccupied ('.')
    # - Seat at index `i+2` is occupied ('#')
    # This corresponds to the problem statement's condition for seat number i+1.
    
    # Using string slicing is a concise way to check the pattern.
    if S[i:i+3] == "#.#":
      count += 1
      
  # Print the final count to standard output.
  print(count)

solve()