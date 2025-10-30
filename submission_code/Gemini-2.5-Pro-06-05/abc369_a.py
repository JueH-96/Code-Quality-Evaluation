import sys

def solve():
  """
  Reads two integers A and B, and calculates the number of integers x
  such that {A, B, x} can form an arithmetic sequence.
  """
  # Read two space-separated integers from a single line of input.
  try:
    A, B = map(int, sys.stdin.readline().split())
  except (IOError, ValueError):
    # This handles potential empty lines or non-integer input,
    # though not expected under problem constraints.
    return

  # Case 1: A and B are the same.
  # If A=B, the only way to form an arithmetic sequence is for all three
  # numbers to be the same (e.g., A, A, A). Thus, x must be equal to A.
  # There is only 1 possible value for x.
  if A == B:
    print(1)
  else:
    # Case 2: A and B are different.
    # We always have two integer solutions for x:
    #   1. Making A the middle term: x = 2*A - B
    #   2. Making B the middle term: x = 2*B - A
    # These two solutions are distinct since A != B.

    # A third potential solution exists if x can be the middle term:
    #   3. x = (A + B) / 2
    # This is an integer solution if and only if (A + B) is even,
    # which means A and B must have the same parity.
    
    # If A and B have the same parity, (A + B) is even, and we get a
    # third distinct integer solution. Total: 3 solutions.
    # If A and B have different parities, (A + B) is odd, so the
    # third potential solution is not an integer. Total: 2 solutions.

    if (A + B) % 2 == 0:
      # A and B have the same parity.
      print(3)
    else:
      # A and B have different parity.
      print(2)

solve()