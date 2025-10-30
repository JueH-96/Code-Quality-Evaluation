# YOUR CODE HERE

import sys

def is_airport_code(S, T):
    # Check if T can be derived from S by taking a subsequence of length 3 and converting it to uppercase
    for i in range(len(S) - 2):
        if T == S[i:i+3].upper():
            return True

    # Check if T can be derived from S by taking a subsequence of length 2, converting it to uppercase, and appending X
    for i in range(len(S) - 1):
        if T == S[i:i+2].upper() + 'X':
            return True

    return False

S = sys.stdin.readline().strip()
T = sys.stdin.readline().strip()

if is_airport_code(S, T):
    print('Yes')
else:
    print('No')