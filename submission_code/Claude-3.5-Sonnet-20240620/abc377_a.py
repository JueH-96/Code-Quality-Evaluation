# YOUR CODE HERE
def can_rearrange_to_abc(s):
    return sorted(s) == sorted('ABC')

# Read input from stdin
S = input().strip()

# Check if it's possible to rearrange S to match 'ABC'
if can_rearrange_to_abc(S):
    print("Yes")
else:
    print("No")