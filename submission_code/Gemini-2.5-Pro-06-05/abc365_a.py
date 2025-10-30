# YOUR CODE HERE
def solve():
  """
  Reads a year Y from standard input and prints the number of days in that year.
  """
  try:
    # Read the year from standard input and convert it to an integer.
    Y = int(input())

    # Apply the rules of the Gregorian calendar to determine the number of days.
    # The conditions are checked in order of precedence.

    # A year is a leap year if it is divisible by 400.
    if Y % 400 == 0:
      print(366)
    # A year is a common year if it is divisible by 100 but not by 400.
    elif Y % 100 == 0:
      print(365)
    # A year is a leap year if it is divisible by 4 but not by 100.
    elif Y % 4 == 0:
      print(366)
    # Otherwise, a year is a common year.
    else:
      print(365)

  except (ValueError, EOFError):
    # Handle potential errors with input, though not expected by the problem constraints.
    pass

solve()