# YOUR CODE HERE
import sys

def main():
  """
  Reads N, L, and R, then generates a sequence from 1 to N.
  It reverses the elements from the L-th to the R-th position
  and prints the resulting sequence.
  """
  try:
    # Read N, L, and R from a single line of standard input.
    # The map() function applies int() to each item of the split input string.
    N, L, R = map(int, sys.stdin.readline().split())
  except (ValueError, IndexError):
    # This block handles potential malformed input, though not expected
    # under the problem constraints.
    return

  # Generate the initial sequence from 1 to N.
  # e.g., for N=5, this creates [1, 2, 3, 4, 5]
  sequence = list(range(1, N + 1))

  # The problem uses 1-based indexing, but Python lists are 0-based.
  # We convert L and R to the appropriate 0-based slice indices.
  # The slice for the L-th to R-th elements is from index (L-1) up to (R).
  start_index = L - 1
  end_index = R

  # Python's slice assignment provides a concise way to reverse a sublist.
  # 1. `sequence[start_index:end_index]` extracts the sublist.
  # 2. `[::-1]` reverses the extracted sublist.
  # 3. The assignment `... = ...` replaces the original slice with the reversed one.
  sequence[start_index:end_index] = sequence[start_index:end_index][::-1]

  # Print the final sequence to standard output.
  # The * operator unpacks the list elements, and print() separates them with spaces.
  print(*sequence)

if __name__ == "__main__":
  main()