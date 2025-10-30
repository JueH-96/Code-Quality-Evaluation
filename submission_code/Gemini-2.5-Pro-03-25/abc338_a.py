# YOUR CODE HERE
import sys

def solve():
    """
    Reads a string S from standard input and determines if it satisfies the condition:
    - The first character is uppercase.
    - All other characters are lowercase.
    Prints "Yes" if the condition is satisfied, otherwise prints "No".
    """
    # Read input string from stdin
    s = sys.stdin.readline().strip()

    # Constraints guarantee 1 <= len(s) <= 100 and s contains only English letters.
    # Therefore, s is non-empty, and accessing s[0] is always safe.

    # Check Condition 1: The first character of S is uppercase.
    # The isupper() method returns True if the character is an uppercase letter, False otherwise.
    first_char_is_upper = s[0].isupper()

    # Check Condition 2: All other characters of S are lowercase.
    # We check characters from the second position (index 1) to the end.
    # The slice s[1:] contains these characters.
    # The all() function takes an iterable (like the generator expression here)
    # and returns True if all elements in the iterable are true.
    # The generator expression (c.islower() for c in s[1:]) yields True for each
    # lowercase character in the rest of the string and False otherwise.
    # If s has only one character, s[1:] is an empty string "".
    # all() applied to an empty iterable returns True. This correctly handles the case
    # where there are no "other characters", satisfying the condition vacuously.
    rest_chars_are_lower = all(c.islower() for c in s[1:])

    # Determine the output based on whether both conditions are satisfied
    if first_char_is_upper and rest_chars_are_lower:
        print("Yes")
    else:
        print("No")

# Execute the solve function when the script is run
if __name__ == '__main__':
    solve()