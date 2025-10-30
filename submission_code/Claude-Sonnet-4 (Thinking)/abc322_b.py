# Read input
N, M = map(int, input().split())
S = input()
T = input()

# Check if S is a prefix of T
is_prefix = (S == T[:N])

# Check if S is a suffix of T
is_suffix = (S == T[-N:])

# Determine output based on conditions
if is_prefix and is_suffix:
    print(0)
elif is_prefix:
    print(1)
elif is_suffix:
    print(2)
else:
    print(3)