# YOUR CODE HERE
def solve():
  """
  Reads the shout time A, bedtime B, and wake-up time C,
  and determines if Takahashi is awake at time A.
  """
  try:
    # Read the three integers from a single line of standard input.
    A, B, C = map(int, input().split())

    # Case 1: The sleeping period does not cross midnight (e.g., bed at 8, wake at 14).
    if B < C:
      # The sleeping interval is from B (inclusive) to C (exclusive).
      # Takahashi is awake if the time A is outside this interval.
      if A >= B and A < C:
        print("No")
      else:
        print("Yes")
    
    # Case 2: The sleeping period crosses midnight (e.g., bed at 21, wake at 7).
    else:  # B > C
      # The sleeping intervals are [B, 24) and [0, C).
      # The awake interval is from C (inclusive) to B (exclusive).
      # Takahashi is awake if the time A is within this interval.
      if A >= C and A < B:
        print("Yes")
      else:
        print("No")

  except (IOError, ValueError):
    # This part handles potential input errors, though not strictly
    # necessary for the constraints of this specific problem.
    pass

solve()