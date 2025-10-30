import sys

# Read N and M
line = sys.stdin.readline().split()
N = int(line[0])
M = int(line[1])

# Read strings S and T
S = sys.stdin.readline().strip()
T = sys.stdin.readline().strip()

# Check if S is a prefix of T
# A prefix of length N is the substring from index 0 up to (but not including) N
# In Python slicing, this is T[:N]
is_prefix = (S == T[:N])

# Check if S is a suffix of T
# A suffix of length N starts at index M - N and goes to the end
# In Python slicing, this is T[M-N:]
is_suffix = (S == T[M-N:])

# Determine the output based on the prefix and suffix checks
if is_prefix and is_suffix:
    # S is both a prefix and a suffix of T
    print(0)
elif is_prefix:
    # S is a prefix of T but not a suffix
    print(1)
elif is_suffix:
    # S is a suffix of T but not a prefix
    print(2)
else:
    # S is neither a prefix nor a suffix of T
    print(3)