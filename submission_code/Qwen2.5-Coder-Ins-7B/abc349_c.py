# YOUR CODE HERE
import sys

def is_airport_code(S, T):
    if len(T) == 3:
        for i in range(len(S) - 2):
            if S[i:i+3].upper() == T:
                return True
    elif len(T) == 4:
        for i in range(len(S) - 2):
            for j in range(i + 1, len(S) - 1):
                if S[i].upper() + S[j].upper() + 'X' == T:
                    return True
    return False

# Read input from stdin
S = sys.stdin.readline().strip()
T = sys.stdin.readline().strip()

# Output the result
if is_airport_code(S, T):
    print("Yes")
else:
    print("No")