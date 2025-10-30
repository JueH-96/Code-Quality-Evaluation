import sys

def is_11_22_string(N, S):
    # Check if the length of the string is odd
    if N % 2 == 0:
        return "No"

    mid = (N + 1) // 2 - 1

    # Check the first part of the string
    for i in range(mid):
        if S[i] != '1':
            return "No"

    # Check the middle character
    if S[mid + 1] != '/':
        return "No"

    # Check the last part of the string
    for i in range(mid + 2, N):
        if S[i] != '2':
            return "No"

    return "Yes"

# Read input from stdin
input = sys.stdin.read
data = input().split()

N = int(data[0])
S = data[1]

# Determine if the string is an 11/22 string
result = is_11_22_string(N, S)

# Write the output to stdout
sys.stdout.write(result + '
')