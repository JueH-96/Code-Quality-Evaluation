# YOUR CODE HERE
N, M = map(int, input().split())
S = input()
T = input()

is_prefix = False
is_suffix = False

# Check if S is a prefix of T
# S is a prefix of T if the first N characters of T coincide with S.
if T[0:N] == S:
    is_prefix = True

# Check if S is a suffix of T
# S is a suffix of T if the last N characters of T coincide with S.
# The last N characters of T can be obtained by slicing from index M-N to M.
if T[M-N:M] == S:
    is_suffix = True

# Apply the conditional logic to print the result
if is_prefix and is_suffix:
    print(0)
elif is_prefix and not is_suffix:
    print(1)
elif not is_prefix and is_suffix:
    print(2)
else: # not is_prefix and not is_suffix
    print(3)