import sys

def solve():
  """
  Reads N strings, sorts them by length, concatenates them, and prints the result.
  """
  
  # Read the number of strings, N.
  # This value is not strictly necessary for the logic if we read all lines,
  # but it's part of the specified input format.
  try:
    N = int(sys.stdin.readline())
  except (ValueError, IndexError):
    # Handle cases with no input
    return

  # Read the N strings into a list.
  # We use a list comprehension for a concise way to build the list.
  # .strip() is used to remove any trailing newline characters from each line.
  strings = [sys.stdin.readline().strip() for _ in range(N)]

  # Sort the list of strings in ascending order of their length.
  # The `key=len` argument tells the sorted() function to use the length
  # of each string as the value for comparison during the sort.
  sorted_strings = sorted(strings, key=len)

  # Join the sorted strings into a single string with no separator.
  # The "".join() method is a highly efficient way to perform string concatenation.
  result = "".join(sorted_strings)

  # Print the final string to standard output.
  print(result)

# Execute the solution
solve()