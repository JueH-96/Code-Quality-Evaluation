# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

S = data[0]
T = data[1]

def is_airport_code(S, T):
    # Check if T can be derived from a subsequence of length 3 in S
    subsequence = []
    for char in S:
        if char.upper() == T[len(subsequence)]:
            subsequence.append(char)
        if len(subsequence) == 3:
            return True

    # Check if T can be derived from a subsequence of length 2 in S followed by 'X'
    subsequence = []
    for char in S:
        if char.upper() == T[len(subsequence)]:
            subsequence.append(char)
        if len(subsequence) == 2:
            if T[2] == 'X':
                return True
            else:
                return False

    return False

if is_airport_code(S, T):
    print("Yes")
else:
    print("No")