import sys

def main():
  """
  Main function to solve the Airport Code problem.
  """
  # Read the input strings from standard input.
  S = sys.stdin.readline().strip()
  T = sys.stdin.readline().strip()

  # Convert T to lowercase for case-insensitive matching.
  t_lower = T.lower()

  def is_subsequence(sub, main):
    """
    Checks if `sub` is a subsequence of `main` using a greedy two-pointer algorithm.
    This is an efficient O(len(main)) approach.
    
    Args:
      sub: The potential subsequence string.
      main: The main string to search within.
      
    Returns:
      True if `sub` is a subsequence of `main`, False otherwise.
    """
    sub_ptr = 0
    main_ptr = 0
    while sub_ptr < len(sub) and main_ptr < len(main):
      if sub[sub_ptr] == main[main_ptr]:
        sub_ptr += 1
      main_ptr += 1
    return sub_ptr == len(sub)

  # Check Condition 1: T is formed from a subsequence of length 3 from S.
  if is_subsequence(t_lower, S):
    print("Yes")
  # Check Condition 2: T is formed from a subsequence of length 2 from S, with 'X' appended.
  # This is only evaluated if Condition 1 is false, and only possible if T ends with 'X'.
  elif T.endswith('X') and is_subsequence(t_lower[:-1], S):
    print("Yes")
  # If neither condition is met, T is not a valid airport code for S.
  else:
    print("No")

if __name__ == "__main__":
  main()