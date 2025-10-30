import sys

def is_airport_code(S, T):
    # Check if T can be formed by taking a subsequence of length 3 from S
    if len(T) == 3 and T != 'XXX':
        i, j, k = 0, 0, 0
        for char in S:
            if char.upper() == T[i]:
                i += 1
            if char.upper() == T[j]:
                j += 1
            if char.upper() == T[k]:
                k += 1
            if i == 3 or j == 3 or k == 3:
                return "Yes"

    # Check if T can be formed by taking a subsequence of length 2 from S and appending 'X'
    if T[-1] == 'X':
        i, j = 0, 0
        for char in S:
            if char.upper() == T[i]:
                i += 1
            if char.upper() == T[j]:
                j += 1
            if i == 2 or j == 2:
                return "Yes"

    return "No"

# Read input from stdin
S = sys.stdin.readline().strip()
T = sys.stdin.readline().strip()

# Output the result to stdout
print(is_airport_code(S, T))