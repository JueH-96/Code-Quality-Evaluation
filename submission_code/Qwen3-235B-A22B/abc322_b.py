# Read input values
n, m = map(int, input().split())
s = input().strip()
t = input().strip()

# Check if S is a prefix and/or suffix of T
is_prefix = t.startswith(s)
is_suffix = t.endswith(s)

# Determine the output based on the conditions
if is_prefix and is_suffix:
    print(0)
elif is_prefix:
    print(1)
elif is_suffix:
    print(2)
else:
    print(3)