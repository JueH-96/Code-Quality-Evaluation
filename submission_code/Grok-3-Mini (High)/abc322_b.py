import sys

# Read all lines from stdin
data = sys.stdin.read().splitlines()

# Parse the first line to get N and M
first_line = data[0].split()
N = int(first_line[0])
M = int(first_line[1])

# Read S and T
S = data[1]
T = data[2]

# Check if S is a prefix of T
is_prefix = (T[:N] == S)

# Check if S is a suffix of T
is_suffix = (T[-N:] == S)

# Determine the output based on conditions
if is_prefix and is_suffix:
    print(0)
elif is_prefix:
    print(1)
elif is_suffix:
    print(2)
else:
    print(3)