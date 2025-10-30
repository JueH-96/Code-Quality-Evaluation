import sys

def solve():
  """
  Reads an integer N and N strings from standard input,
  then counts and prints the number of strings equal to "Takahashi".
  """
  try:
    # Read the number of strings
    N = int(sys.stdin.readline())
  except (ValueError, IndexError):
    # This case should not happen based on constraints, but is good practice
    return

  # Initialize a counter for the name "Takahashi"
  takahashi_count = 0

  # Loop N times to read each string
  for _ in range(N):
    # Read a string and remove any trailing newline characters
    s = sys.stdin.readline().strip()
    
    # Check if the string is "Takahashi"
    if s == "Takahashi":
      # If it is, increment our counter
      takahashi_count += 1
  
  # Print the final count
  print(takahashi_count)

solve()