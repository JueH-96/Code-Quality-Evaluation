import sys

# Read input from stdin
data = sys.stdin.read().split()
N = int(data[0])
S = data[1]
T = data[2]

# Define a function to check if two characters are similar
def are_similar(c1, c2):
    if c1 == c2:
        return True
    if (c1 == '1' and c2 == 'l') or (c1 == 'l' and c2 == '1'):
        return True
    if (c1 == '0' and c2 == 'o') or (c1 == 'o' and c2 == '0'):
        return True
    return False

# Check if all corresponding characters are similar
if all(are_similar(S[i], T[i]) for i in range(N)):
    print("Yes")
else:
    print("No")