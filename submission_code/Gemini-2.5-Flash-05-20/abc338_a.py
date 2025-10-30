import sys

def solve():
    S = sys.stdin.readline().strip()

    # Condition 1: The first character of S must be uppercase.
    # S is guaranteed to be non-empty due to constraints (1 <= |S|).
    is_first_char_uppercase = S[0].isupper()

    # Condition 2: All other characters (from the second character to the end) must be lowercase.
    # This applies to the substring S[1:].
    # If S has only one character (e.g., S = "A"), then S[1:] will be an empty string.
    # The `all()` function, when applied to an empty iterable, returns True,
    # which correctly handles the single-character case (e.g., "A" should be "Yes").
    are_other_chars_lowercase = all(char.islower() for char in S[1:])

    # Both conditions must be satisfied.
    if is_first_char_uppercase and are_other_chars_lowercase:
        sys.stdout.write("Yes
")
    else:
        sys.stdout.write("No
")

# Call the solve function to execute the program
if __name__ == '__main__':
    solve()