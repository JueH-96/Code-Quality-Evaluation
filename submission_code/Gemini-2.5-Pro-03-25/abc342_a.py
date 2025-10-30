# YOUR CODE HERE
import sys

# Define the main function to solve the problem
def solve():
    """
    Reads a string S from stdin, finds the 1-based index of the character
    that is different from all other characters, and prints the index to stdout.

    Constraints:
    - Length of S is between 3 and 100, inclusive.
    - S consists of exactly two different lowercase English letters.
    - All characters but one of S are the same.
    """
    
    # Read the input string from standard input and remove leading/trailing whitespace
    s = sys.stdin.readline().strip()
    # Get the length of the string
    n = len(s)

    # Logic to find the unique character's index:
    
    # Since the length N is guaranteed to be at least 3, we can safely access
    # s[0], s[1], and s[2].

    # Case 1: The first two characters are identical.
    if s[0] == s[1]:
        # If the first two characters are the same, this character must be the common one,
        # because only one character in the entire string is different.
        common_char = s[0]
        
        # We need to find the index 'i' where s[i] is different from the common character.
        # We can start checking from the third character (index 2).
        for i in range(2, n):
            # If we find a character that is different from the common character
            if s[i] != common_char:
                # This character at index 'i' is the unique one.
                # Print its 1-based index (i + 1) and terminate.
                print(i + 1)
                return # Exit the function as the answer is found

    # Case 2: The first two characters are different (s[0] != s[1]).
    else:
        # If the first two characters are different, one of them must be the unique character,
        # and the other must be the common character.
        # We can determine which is which by looking at the third character, s[2].

        if s[0] == s[2]:
            # If the first character (s[0]) matches the third character (s[2]),
            # then s[0] must be the common character (since it appears at least twice).
            # Therefore, the second character (s[1]) must be the unique one.
            # The 1-based index of s[1] is 2.
            print(2)
        else:
            # If the first character (s[0]) does not match the third character (s[2]),
            # given that s[0] is also different from s[1],
            # it logically follows that s[1] must match s[2] (because only one character is unique).
            # This means s[1] (and s[2]) represents the common character.
            # Therefore, the first character (s[0]) must be the unique one.
            # The 1-based index of s[0] is 1.
            print(1)

# Call the solve function to execute the logic when the script is run
solve()

# YOUR CODE HERE