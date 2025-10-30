# Read N and M
N, M = map(int, input().strip().split())

# Read S
S = input().strip()

# Read T
T = input().strip()

# Check if S is a prefix of T
is_prefix = S == T[:N]

# Check if S is a suffix of T
is_suffix = S == T[-N:]

# Decide what to print based on the checks
if is_prefix and is_suffix:
    print(0)
elif is_prefix:
    print(1)
elif is_suffix:
    print(2)
else:
    print(3)