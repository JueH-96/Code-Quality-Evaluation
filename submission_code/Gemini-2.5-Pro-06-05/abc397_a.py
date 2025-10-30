# YOUR CODE HERE
import sys

def main():
  """
  Reads a body temperature and classifies it according to the problem specification.
  """
  try:
    # Read the input from standard input
    # The temperature X is given as a string, possibly with a decimal point
    x_str = sys.stdin.readline().strip()

    # Convert the string to a floating-point number
    x = float(x_str)

    # Apply the classification logic using if-elif-else
    # The conditions are checked in order.
    # 1. Check for High fever (>= 38.0)
    if x >= 38.0:
      print(1)
    # 2. If not High fever, check for Fever (>= 37.5)
    #    This branch is only reached if x < 38.0
    elif x >= 37.5:
      print(2)
    # 3. If neither of the above, it must be Normal (< 37.5)
    else:
      print(3)

  except (ValueError, IndexError):
    # Handle potential errors if the input is not a valid number,
    # though the problem constraints suggest this won't happen.
    # This is good practice for robust code.
    pass

if __name__ == "__main__":
  main()