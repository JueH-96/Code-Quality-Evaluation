import sys

def normalize_char(char_val):
    """
    Normalizes a character to its canonical form for similarity comparison.
    - '1' and 'l' are mapped to 'l'.
    - '0' and 'o' are mapped to 'o'.
    - Other characters remain unchanged.
    """
    if char_val == '1' or char_val == 'l':
        return 'l'
    elif char_val == '0' or char_val == 'o':
        return 'o'
    else:
        return char_val

def are_chars_similar(char1, char2):
    """
    Checks if two characters are similar based on the given rules.
    This is achieved by comparing their normalized forms.
    """
    return normalize_char(char1) == normalize_char(char2)

def solve():
    """
    Reads the input strings S and T, and determines if they are similar.
    Prints "Yes" if they are similar, "No" otherwise.
    """
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    is_similar_strings = True
    for i in range(N):
        # If any pair of characters at the same position is not similar,
        # then the strings are not similar.
        if not are_chars_similar(S[i], T[i]):
            is_similar_strings = False
            break # No need to check further, we already know the answer

    if is_similar_strings:
        sys.stdout.write("Yes
")
    else:
        sys.stdout.write("No
")

# Call the solve function to run the program
solve()