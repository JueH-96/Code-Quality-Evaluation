import sys

# Read input from stdin
N, M = map(int, input().split())
S = input().strip()
T = input().strip()

# Check if S is a prefix of T
is_prefix = T.startswith(S)
# Check if S is a suffix of T
is_suffix = T.endswith(S)

# Determine the output based on the conditions
if is_prefix and is_suffix:
    print(0)
elif is_prefix:
    print(1)
elif is_suffix:
    print(2)
else:
    print(3)