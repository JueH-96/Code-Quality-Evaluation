# YOUR CODE HERE
import sys

def is_extended_abc(s):
    if not s:
        return True
    if s[0] == 'A':
        return is_extended_abc(s[1:]) and s[1:].startswith('B') and is_extended_abc(s[1:].replace('B', ''))
    elif s[0] == 'B':
        return is_extended_abc(s[1:]) and s[1:].startswith('C') and is_extended_abc(s[1:].replace('C', ''))
    elif s[0] == 'C':
        return is_extended_abc(s[1:])
    return False

# Read input from stdin
s = sys.stdin.read().strip()

# Check if the string is an Extended ABC string
if is_extended_abc(s):
    print("Yes")
else:
    print("No")