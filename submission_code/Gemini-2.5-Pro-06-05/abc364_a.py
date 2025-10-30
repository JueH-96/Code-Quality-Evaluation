import sys

def solve():
  """
  This function reads the input, determines if all dishes can be eaten,
  and prints the result.
  """
  try:
    n = int(sys.stdin.readline())
    # Handle the case of no dishes being provided after N, e.g., on an empty input stream
    if n == 0:
        print("Yes")
        return
    dishes = [sys.stdin.readline().strip() for _ in range(n)]
  except (IOError, ValueError):
    # In case of malformed input or empty stream, handle gracefully.
    # For this problem's constraints, this is not strictly necessary but good practice.
    # An empty dishes list would mean N=0, which is handled above.
    return
  
  # Assume Takahashi can eat all dishes unless a failure condition is met.
  can_eat_all = "Yes"

  # We only fail if two consecutive sweet dishes appear *before* the last dish.
  # Eating the last two dishes as 'sweet' is not a failure because there are no more dishes to eat.
  #
  # A failure occurs if we find a pair (dishes[i], dishes[i+1]) that are both 'sweet',
  # and there is at least one more dish, dishes[i+2], to eat.
  # This means the index i+1 must be less than n-1, which simplifies to i < n-2.
  # So, we need to loop with i from 0 up to n-3.
  # The Python `range(n - 2)` generates indices from 0 to n-3, which is exactly what we need.
  # If n <= 2, this range is empty, the loop is skipped, and "Yes" is printed, which is correct.
  for i in range(n - 1):
    if dishes[i] == 'sweet' and dishes[i+1] == 'sweet':
      # If this consecutive sweet pair occurs before the final dish, it's a failure.
      # The second sweet dish is at index i+1. It's a failure if i+1 is not the last index (n-1).
      if i + 1 < n - 1:
        can_eat_all = "No"
        break
  
  print(can_eat_all)

solve()