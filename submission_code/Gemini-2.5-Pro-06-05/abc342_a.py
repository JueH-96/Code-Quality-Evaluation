# YOUR CODE HERE
def solve():
  """
  Reads a string S and finds the 1-based index of the character
  that differs from all others.
  """
  S = input()

  # Case 1: The first two characters are the same.
  # This implies they are the "common" character.
  # We must find the different character in the rest of the string.
  if S[0] == S[1]:
    common_char = S[0]
    # We can start checking from the third character (index 2).
    for i in range(2, len(S)):
      if S[i] != common_char:
        # Found the unique character. Print its 1-based index.
        print(i + 1)
        return

  # Case 2: The first two characters are different.
  # One of them is unique. S[2] must be the common character.
  # If S[0] and S[2] are the same, S[1] must be the unique one.
  elif S[0] == S[2]:
    # The 1-based index of S[1] is 2.
    print(2)
    return

  # Case 3: S[0] != S[1] and S[0] != S[2].
  # This implies S[1] and S[2] are the common characters,
  # so S[0] must be the unique one.
  else:
    # The 1-based index of S[0] is 1.
    print(1)
    return

solve()