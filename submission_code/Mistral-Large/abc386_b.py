import sys

def min_button_presses(S):
    n = len(S)
    count = 0
    i = 0

    while i < n:
        if i < n - 1 and S[i] == '0' and S[i + 1] == '0':
            count += 1
            i += 2
        else:
            count += 1
            i += 1

    return count

# Read input from stdin
S = sys.stdin.read().strip()

# Calculate the minimum number of button presses
result = min_button_presses(S)

# Write the output to stdout
sys.stdout.write(str(result) + '
')