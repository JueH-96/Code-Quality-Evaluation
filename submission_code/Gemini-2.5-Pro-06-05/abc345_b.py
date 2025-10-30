# YOUR CODE HERE
import sys

def solve():
  """
  Reads an integer X from standard input, calculates ceil(X/10),
  and prints the result to standard output.
  """
  # Read the integer from standard input. Python's int type handles
  # large numbers like 10^18 automatically.
  X = int(sys.stdin.readline())

  # To calculate ceil(X / 10) using integer arithmetic, we can use the
  # identity ceil(a) = -floor(-a). In Python, floor division is
  # performed by the `//` operator.
  # So, ceil(X / 10) is equivalent to -(-X // 10).
  # This avoids floating-point issues and works for all integers.
  result = -(-X // 10)
  
  print(result)

solve()