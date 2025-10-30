# YOUR CODE HERE
import sys

# Read N
N = int(sys.stdin.readline())

# Read S
S = sys.stdin.readline().strip()

# Check if S is an 11/22 string based on the conditions:

# Condition 1: |S| is odd.
if N % 2 == 0:
    print("No")
else:
    # Calculate the middle index (0-based)
    # For odd N, (N+1)/2 is the 1-based index of the middle character.
    # The 0-based index is (N+1)//2 - 1 = N//2.
    mid = N // 2

    # Condition 3: The middle character is '/'.
    # This is the character at 0-based index 'mid'.
    if S[mid] != '/':
        print("No")
    else:
        # Condition 2: The prefix characters are all '1'.
        # The prefix consists of characters from 1-based index 1 up to (N+1)/2 - 1.
        # In 0-based index, this is from index 0 up to (N+1)//2 - 2 = mid - 1.
        # This corresponds to the slice S[0:mid].
        actual_prefix = S[0:mid]
        expected_prefix = '1' * mid # A string of '1's with length 'mid'.

        # Condition 4: The suffix characters are all '2'.
        # The suffix consists of characters from 1-based index (N+1)/2 + 1 up to N.
        # In 0-based index, this is from index (N+1)//2 up to N - 1.
        # Since (N+1)//2 = mid + 1 for odd N, this is from index mid + 1 up to N - 1.
        # This corresponds to the slice S[mid+1:N].
        actual_suffix = S[mid+1:N]
        # The length of the suffix is N - (mid + 1).
        # Since N is odd, N = 2*k + 1. mid = k. Length = (2*k + 1) - (k + 1) = k = mid.
        # The suffix should be a string of '2's with length 'mid'.
        expected_suffix = '2' * mid # A string of '2's with length 'mid'.

        # Check if the actual prefix and suffix match the expected ones.
        if actual_prefix == expected_prefix and actual_suffix == expected_suffix:
            print("Yes")
        else:
            print("No")