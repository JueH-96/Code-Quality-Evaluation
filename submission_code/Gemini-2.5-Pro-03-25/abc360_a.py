# YOUR CODE HERE
import sys

def solve():
    """
    Reads the arrangement string S from standard input.
    The string S represents three plates: Rice ('R'), Miso soup ('M'), and Salad ('S').
    Determines if the plate of rice ('R') is positioned to the left of
    the plate of miso soup ('M').
    Prints "Yes" to standard output if 'R' is to the left of 'M',
    and "No" otherwise.
    """
    # Read the single line of input representing the plate arrangement
    # .strip() removes any leading/trailing whitespace, including the newline character
    s = sys.stdin.readline().strip()

    # Use the string find() method to get the 0-based index of the first occurrence of 'R'.
    # Since the constraints state there is exactly one 'R', this gives its position.
    r_index = s.find('R')

    # Similarly, find the 0-based index of the first occurrence of 'M'.
    # There is exactly one 'M'.
    m_index = s.find('M')

    # Compare the indices (positions). If the index of 'R' is less than the index of 'M',
    # it means 'R' appears earlier in the string, i.e., it is to the left of 'M'.
    if r_index < m_index:
        # If 'R' is to the left of 'M', print "Yes"
        print("Yes")
    else:
        # Otherwise (if 'R' is at the same position or to the right of 'M'), print "No".
        # Since 'R' and 'M' are distinct characters in a string of unique characters R, M, S,
        # they cannot be at the same position. So this 'else' covers the case where r_index > m_index.
        print("No")

# Call the main function to execute the logic when the script is run
solve()

# END YOUR CODE HERE