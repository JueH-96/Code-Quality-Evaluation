# YOUR CODE HERE
def solve():
  """
  Solves the pen buying problem.
  Reads the costs of red, green, and blue pens, and the disliked color.
  Then, it calculates and prints the minimum cost to buy one pen,
  avoiding the disliked color.
  """
  # Read the costs of the pens from the first line of input.
  # The input is space-separated, so we split the string and map each part to an integer.
  r, g, b = map(int, input().split())

  # Read the disliked color from the second line of input.
  c = input()

  # Determine the available options based on the disliked color and find the minimum cost.
  if c == "Red":
    # If Takahashi dislikes Red, he can buy a Green or a Blue pen.
    # The minimum cost is the lesser of the Green and Blue pen costs.
    result = min(g, b)
  elif c == "Green":
    # If Takahashi dislikes Green, he can buy a Red or a Blue pen.
    # The minimum cost is the lesser of the Red and Blue pen costs.
    result = min(r, b)
  else: # c must be "Blue" based on the problem constraints
    # If Takahashi dislikes Blue, he can buy a Red or a Green pen.
    # The minimum cost is the lesser of the Red and Green pen costs.
    result = min(r, g)

  # Print the calculated minimum cost to standard output.
  print(result)

# Call the solve function to run the program.
solve()