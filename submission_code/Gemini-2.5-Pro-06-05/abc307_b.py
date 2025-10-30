import sys

def main():
  """
  Reads N strings and determines if any two distinct strings, S_i and S_j,
  can be concatenated to form a palindrome.
  """
  try:
    # Read the number of strings.
    N = int(sys.stdin.readline())
    
    # Read N strings into a list.
    S = [sys.stdin.readline().strip() for _ in range(N)]
  except (ValueError, IndexError):
    # This handles potential malformed input, though problem constraints
    # make this unlikely.
    # In a contest setting, such robust error handling might be omitted
    # for brevity.
    return

  # Iterate through all possible pairs of distinct indices (i, j).
  for i in range(N):
    for j in range(N):
      # The problem specifies that integers i and j must be distinct.
      if i == j:
        continue
      
      # Concatenate the two strings in the order S_i, S_j.
      concatenated_str = S[i] + S[j]
      
      # Check if the resulting string is a palindrome.
      # A string is a palindrome if it reads the same forwards and backwards.
      # In Python, this is easily checked by comparing the string to its reverse.
      if concatenated_str == concatenated_str[::-1]:
        # If a palindrome is found, print "Yes" and terminate.
        print("Yes")
        return
        
  # If the loops complete without finding a palindrome, no such pair exists.
  print("No")

if __name__ == "__main__":
  main()