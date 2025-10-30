# YOUR CODE HERE
import sys

def solve():
    """
    Reads a string, counts uppercase and lowercase letters,
    and converts the string based on which count is greater.
    """
    # Read the string from standard input.
    S = sys.stdin.readline().strip()

    # Count the number of uppercase characters using a generator expression.
    # This is a concise and efficient way to count items matching a condition.
    upper_count = sum(1 for char in S if char.isupper())

    # The number of lowercase characters is the total length minus the uppercase count,
    # as the string is guaranteed to contain only English letters.
    lower_count = len(S) - upper_count

    # Compare the counts and perform the appropriate conversion.
    if upper_count > lower_count:
        # If uppercase letters are more numerous, convert the whole string to uppercase.
        print(S.upper())
    else:
        # Otherwise, convert the whole string to lowercase.
        # The case where counts are equal is impossible since the string length is odd.
        print(S.lower())

solve()