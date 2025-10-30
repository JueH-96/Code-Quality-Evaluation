# YOUR CODE HERE
import sys

def solve():
    """
    Reads input N and S, checks if S is an 11/22 string according to the definition,
    and prints "Yes" or "No".
    """
    # Read input N (length of the string)
    n = int(sys.stdin.readline())
    # Read input string S
    s = sys.stdin.readline().strip()

    # Check if S is an 11/22 string based on the definition conditions.

    # Condition 1: The length N must be odd.
    # If N is even, it cannot be an 11/22 string.
    if n % 2 == 0:
        print("No")
        return # Exit the function if the length is even.

    # If N is odd, proceed with other checks.
    # Calculate the 0-based index of the middle character.
    # Since N is odd, let N = 2k + 1 for some integer k >= 0.
    # The 1-based middle position is (N+1)/2 = (2k+1+1)/2 = (2k+2)/2 = k+1.
    # The 0-based middle index is (k+1) - 1 = k.
    # Also, k = (N - 1) / 2. In integer division, this is N // 2.
    mid_index = n // 2

    # Condition 3: The middle character must be '/'.
    # The character at the middle index must be '/'.
    # We check this before checking the parts before and after the slash.
    # Since N >= 1 and N is odd, mid_index is a valid index (0 <= mid_index < n).
    if s[mid_index] != '/':
        print("No")
        return # Exit if the middle character is not '/'.

    # Condition 2: Characters before the middle '/' must all be '1'.
    # These characters are at indices 0 to mid_index - 1.
    # The number of characters in this part is mid_index.
    # We create the expected string of '1's and compare it with the slice.
    # If n=1, mid_index=0, the slice s[0:0] is "", and '1'*0 is "". They match.
    expected_first_part = '1' * mid_index
    if s[0:mid_index] != expected_first_part:
        print("No")
        return # Exit if the first part is not all '1's.

    # Condition 4: Characters after the middle '/' must all be '2'.
    # These characters are at indices mid_index + 1 to n - 1.
    # The number of characters in this part is (n - 1) - (mid_index + 1) + 1 = n - 1 - mid_index.
    # Since n = 2 * mid_index + 1, the length is (2 * mid_index + 1) - 1 - mid_index = mid_index.
    # We create the expected string of '2's and compare it with the slice.
    # If n=1, mid_index=0, the slice s[1:1] is "", and '2'*0 is "". They match.
    expected_second_part = '2' * mid_index
    if s[mid_index + 1 : n] != expected_second_part:
        print("No")
        return # Exit if the second part is not all '2's.

    # If all conditions were met and the function did not return "No" early,
    # then the string S is an 11/22 string.
    print("Yes")

# Call the solve function to execute the logic based on standard input
solve()