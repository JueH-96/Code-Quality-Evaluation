# YOUR CODE HERE
N, M = map(int, input().split())
S = input().strip()
T = input().strip()

# Check if S is a prefix of T
is_prefix = T[:N] == S

# Check if S is a suffix of T
is_suffix = T[-N:] == S

if is_prefix and is_suffix:
    print(0)
elif is_prefix and not is_suffix:
    print(1)
elif not is_prefix and is_suffix:
    print(2)
else:
    print(3)