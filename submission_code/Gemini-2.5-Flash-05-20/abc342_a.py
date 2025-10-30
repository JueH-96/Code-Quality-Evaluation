S = input()

# The problem states that the string S consists of lowercase English letters,
# has a length between 3 and 100, inclusive, and all characters but one are the same.
# We need to find the 1-based index of the character that differs from all others.

# Since the string length is at least 3, we can always safely access S[0], S[1], and S[2].
# We use these first three characters to determine the "dominant" character (the one that appears N-1 times).

# Case 1: The first two characters are the same (S[0] == S[1]).
# This means S[0] (and S[1]) must be the dominant character.
# We then iterate from the third character (index 2) onwards to find the character
# that is different from this dominant character.
if S[0] == S[1]:
    dominant_char = S[0]
    # Iterate from the third character to the end of the string.
    for i in range(2, len(S)):
        if S[i] != dominant_char:
            # Found the differing character. Print its 1-based index and exit.
            print(i + 1)
            break
# Case 2: The first two characters are different (S[0] != S[1]).
# This implies that one of S[0] or S[1] is the unique character, and the other
# is the dominant character. We can determine which is which by checking S[2].
else: # S[0] != S[1]
    if S[0] == S[2]:
        # If S[0] is the same as S[2], then S[0] must be the dominant character.
        # This means S[1] is the unique character.
        # S[1] is at 0-based index 1, so its 1-based index is 2.
        print(2)
    else: # If S[0] != S[2], then S[1] must be the same as S[2]
          # (because there's only one differing character in the whole string).
          # Therefore, S[1] is the dominant character, and S[0] is the unique character.
          # S[0] is at 0-based index 0, so its 1-based index is 1.
        print(1)