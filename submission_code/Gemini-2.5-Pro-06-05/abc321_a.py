# YOUR CODE HERE
def solve():
  """
  Reads an integer N from standard input and determines if it is a 321-like Number.
  A 321-like Number has digits that are strictly decreasing from left to right.
  """
  
  # Read the number as a string to easily access its digits.
  n_str = input()
  
  # The all() function checks if all elements in an iterable are True.
  # We use a generator expression to create an iterable of boolean values.
  # For each pair of adjacent digits n_str[i] and n_str[i+1], we check if
  # the left digit is strictly greater than the right one.
  # Character comparison ('3' > '2') works correctly for digits '0'-'9'.
  #
  # This approach correctly handles single-digit numbers. For a single digit,
  # len(n_str) is 1, so range(len(n_str) - 1) becomes range(0), which is empty.
  # The all() function on an empty iterable returns True, as required.
  is_321_like = all(n_str[i] > n_str[i+1] for i in range(len(n_str) - 1))

  # Print the final result.
  if is_321_like:
    print("Yes")
  else:
    print("No")

solve()