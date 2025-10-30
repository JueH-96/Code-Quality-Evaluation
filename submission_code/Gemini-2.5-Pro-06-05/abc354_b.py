# YOUR CODE HERE
import sys

def solve():
  """
  This function solves the AtCoder RPS 2 problem.
  """
  try:
    # Read the number of users
    n_str = sys.stdin.readline()
    if not n_str:
      return
    n = int(n_str)
    
    users = []
    total_rating = 0
    
    # Read each user's data
    for _ in range(n):
      line = sys.stdin.readline()
      if not line:
        break
      s, c_str = line.split()
      c = int(c_str)
      
      # Store user data as a tuple (name, rating)
      users.append((s, c))
      # Accumulate the total rating
      total_rating += c
      
    # Sort the users lexicographically by their names.
    # Python's default sort for tuples works by comparing elements from left to right,
    # so sorting the list of (name, rating) tuples will sort them by name.
    users.sort()
    
    # Calculate the winner's index according to the rule T mod N
    winner_index = total_rating % n
    
    # Get the winner's name from the sorted list
    winner_name = users[winner_index][0]
    
    # Print the winner's name
    print(winner_name)

  except (IOError, ValueError) as e:
    # Handle potential input errors, though not expected with problem constraints
    pass

solve()