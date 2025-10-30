# YOUR CODE HERE
def solve():
  """
  Reads the costs of a bat and a glove and prints the name of the more expensive item.
  """
  # Read the two integer inputs from a single line, separated by a space.
  # map(int, input().split()) reads the line, splits it by space, and converts each part to an integer.
  b, g = map(int, input().split())

  # Compare the cost of the bat (b) with the cost of the glove (g).
  # The problem guarantees that b is not equal to g.
  if b > g:
    # If the bat is more expensive, print "Bat".
    print("Bat")
  else:
    # If the glove is more expensive, print "Glove".
    print("Glove")

# Call the solve function to run the program.
solve()