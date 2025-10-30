# YOUR CODE HERE
def solve():
  """
  This function reads an integer N and a list of N distinct integers A.
  It then finds and prints the 1-based index of the second largest element in A.
  """
  
  # Read the number of elements, N.
  try:
    N = int(input())
  except (ValueError, EOFError):
    return # Gracefully exit on bad or empty input

  # Read the list of N integers, A.
  try:
    A = list(map(int, input().split()))
  except (ValueError, EOFError):
    return # Gracefully exit on bad or empty input

  # We need to find the second largest number and its original position.
  # A good way to do this is to pair each number with its original index.
  # We use 1-based indexing as required by the problem statement.
  # enumerate(A, 1) generates pairs of (1-based index, value).
  # We create a list of (value, index) tuples to sort by value.
  indexed_A = []
  for index, value in enumerate(A, 1):
    indexed_A.append((value, index))

  # Sort the list of tuples. By default, Python sorts tuples by their first
  # element. We sort in descending order to get the largest values first.
  # The `key=lambda x: x[0]` explicitly tells sort to use the first element
  # of the tuple (the value) for comparison.
  indexed_A.sort(key=lambda x: x[0], reverse=True)

  # After sorting, the second element in the list (at index 1) is the
  # tuple corresponding to the second largest value.
  # The original index is the second element of this tuple.
  second_largest_index = indexed_A[1][1]

  # Print the final result.
  print(second_largest_index)

# Run the solution
solve()