import sys

def solve():
  """
  Reads a string S from standard input and computes the minimum number of insertions
  to make it a valid alternating string of the form "ioio...".
  """
  # Read the input string from standard input.
  S = sys.stdin.readline().strip()
  n = len(S)

  # s_ptr points to the current character in S.
  s_ptr = 0
  # t_ptr points to the current position in the target "ioio..." string.
  # It represents the length of the prefix of the target string constructed so far.
  t_ptr = 0

  # Greedily build the shortest "ioio..." prefix that contains S as a subsequence.
  while s_ptr < n:
    # Determine the character expected at the current position of the target string.
    expected_char = 'i' if t_ptr % 2 == 0 else 'o'

    # If the current character in S matches the expected character, we can use it.
    if S[s_ptr] == expected_char:
      # We have "placed" this character from S, so we move to the next one.
      s_ptr += 1
    
    # The position in the target string always advances.
    # If it was a match, the character from S filled this position.
    # If it was a mismatch, we must insert the expected_char to fill this position.
    t_ptr += 1

  # t_ptr now holds the length of the shortest "ioio..." prefix that contains S.
  # The final string must have an even length.
  final_t_len = t_ptr
  if final_t_len % 2 != 0:
    # If the length is odd, we need one more character (and thus one more insertion)
    # to make the total length even.
    final_t_len += 1

  # The total number of insertions is the length of the final valid string
  # minus the length of the original string.
  insertions = final_t_len - n

  # Print the result.
  print(insertions)

solve()