import sys

def solve():
  """
  This function solves the problem by finding the indices of correctly typed characters.
  It reads two strings, S (the intended string) and T (the actual typed string),
  from standard input. It then identifies the positions in T that correspond to the
  characters of S, assuming S is a subsequence of T formed by correct key presses.
  """
  
  # Read the target string S and the typed string T from standard input.
  # Using sys.stdin.readline().strip() is efficient for large inputs.
  try:
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
  except (IOError, IndexError):
    # This handles cases where input might be empty, though problem constraints
    # guarantee non-empty strings.
    return

  # s_ptr is the index for the character we are currently looking for in S.
  s_ptr = 0
  
  # result_indices will store the 1-based indices of the correctly typed characters.
  result_indices = []

  # Iterate through the typed string T using its index t_ptr and character char_t.
  for t_ptr, char_t in enumerate(T):
    # We only need to search if we haven't found all characters of S yet.
    # This also prevents an IndexError on S[s_ptr].
    if s_ptr < len(S) and S[s_ptr] == char_t:
      # If the character in T matches the target character from S, it's a correct keystroke.
      # Append its 1-based index to our results.
      result_indices.append(t_ptr + 1)
      # Move on to look for the next character in S.
      s_ptr += 1

  # Print the list of indices, separated by spaces.
  # The '*' operator unpacks the list elements as arguments to the print function.
  print(*result_indices)

# Execute the main logic.
solve()