# YOUR CODE HERE
import sys

def solve():
  """
  Reads integers from standard input until a 0 is encountered,
  then prints them in reverse order, one per line.
  """
  
  # A list to store the numbers read from input.
  numbers = []
  
  # Loop indefinitely to read numbers line by line.
  while True:
    # Read a line from standard input and convert it to an integer.
    # The input() function reads a line and strips the trailing newline.
    # int() converts the resulting string to an integer.
    try:
      n = int(input())
    except EOFError:
      # If the input stream ends before a 0 is found, stop reading.
      # This makes the code robust, although the problem guarantees a 0 terminator.
      break
      
    # Add the newly read number to our list.
    numbers.append(n)
    
    # The problem specifies that the input sequence is terminated by 0.
    # If the number read is 0, we exit the loop.
    if n == 0:
      break
      
  # The `reversed()` built-in function returns a memory-efficient reverse iterator.
  # We loop through this iterator to print each number.
  for num in reversed(numbers):
    # The print() function automatically adds a newline after its argument,
    # which matches the required output format.
    print(num)

# Execute the main logic.
solve()