# YOUR CODE HERE
def solve():
  """
  Reads a graph's adjacency matrix and prints its adjacency list.
  """
  try:
    # Read the number of vertices, N.
    N_str = input()
    if not N_str: return
    N = int(N_str)
  except (EOFError, ValueError):
    return

  # The problem requires N lines of output, one for each vertex from 1 to N.
  # The input provides N rows of an adjacency matrix. The i-th row of the input
  # corresponds to the connections of vertex i. We can process each row as it's read.
  for _ in range(N):
    try:
      # Read a row of the adjacency matrix. This corresponds to the connections
      # of the current vertex we are processing.
      row_str = input()
      if not row_str:
          print() # Handle empty lines for robustness, print newline.
          continue
      row = list(map(int, row_str.split()))
    except (EOFError, ValueError):
      break

    # This list will hold the 1-based vertex numbers of the neighbors.
    neighbors = []

    # Iterate through the row using enumerate to get both the 0-based index and the value.
    # The column index 'j' corresponds to a potential connection to vertex 'j + 1'.
    for j, is_connected in enumerate(row):
      # An edge exists if the value in the adjacency matrix is 1.
      if is_connected == 1:
        # Add the neighbor's 1-based vertex number to the list.
        neighbors.append(j + 1)

    # The 'neighbors' list is naturally in ascending order because we iterate
    # through the columns from left to right (j increases from 0 to N-1).

    # Print the list of neighbors, separated by spaces.
    # The * operator unpacks the list elements as arguments to print().
    # If the list is empty, this correctly prints a blank line.
    print(*neighbors)

solve()