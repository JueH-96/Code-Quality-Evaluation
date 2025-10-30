# YOUR CODE HERE
def solve():
  """
  Solves the card stack problem by simulating stack operations using a Python list.
  """
  
  # Initialize the stack with 100 cards, each labeled 0.
  # A Python list is used to represent the stack.
  # list.append() acts as a push operation.
  # list.pop() acts as a pop operation.
  stack = [0] * 100

  # Read the number of queries from standard input.
  try:
    Q = int(input())
  except (ValueError, EOFError):
    # In case of no input or invalid format for Q, exit gracefully.
    return

  # Process all Q queries.
  for _ in range(Q):
    # Read a single query line and split it into its components.
    # This correctly handles both "1 x" and "2" formats.
    query = input().split()
    
    # The first component is always the query type.
    query_type = int(query[0])
    
    if query_type == 1:
      # Type 1: Place a card with value x on top of the stack (push).
      value = int(query[1])
      stack.append(value)
    else:  # query_type must be 2
      # Type 2: Remove the top card and print its value (pop).
      # The problem guarantees the stack will not be empty, so we can
      # safely call pop() without checking if the stack is empty.
      removed_card = stack.pop()
      print(removed_card)

# Execute the solution
solve()