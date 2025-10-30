# YOUR CODE HERE
import sys

def solve():
    """
    Reads N and S from standard input, determines if S is an 11/22 string,
    and prints the result to standard output.
    """
    try:
        # Read the length of the string from the first line of input.
        N = int(sys.stdin.readline())
        # Read the string from the second line and remove trailing whitespace.
        S = sys.stdin.readline().strip()
    except (ValueError, IndexError):
        # Handle cases of malformed or empty input.
        # Such input cannot form a valid 11/22 string.
        print("No")
        return

    # Initialize the answer to "No". It will be changed to "Yes" only if
    # the string S meets all the required conditions.
    answer = "No"

    # First condition: The length N of an 11/22 string must be odd.
    if N % 2 == 1:
        # For a given odd length N, there is only one possible candidate for
        # an 11/22 string. We can construct this string and compare it.

        # Calculate 'k', which is the number of '1's and '2's on each side
        # of the central '/'. For an odd length N, k = (N - 1) / 2.
        k = (N - 1) // 2

        # Construct the expected 11/22 string of length N.
        # It consists of k '1's, a central '/', and k '2's.
        expected_string = '1' * k + '/' + '2' * k

        # The remaining conditions (first part '1's, middle '/', second part '2's)
        # are all checked by this single comparison.
        if S == expected_string:
            answer = "Yes"

    # Print the final determined answer.
    print(answer)

# Execute the solution.
solve()