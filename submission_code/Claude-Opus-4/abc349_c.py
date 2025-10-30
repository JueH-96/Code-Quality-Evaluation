# YOUR CODE HERE
def is_subsequence(s, t):
    """Check if t is a subsequence of s (case-insensitive)"""
    i = 0
    for char in s:
        if i < len(t) and char.upper() == t[i]:
            i += 1
    return i == len(t)

# Read input
S = input().strip()
T = input().strip()

# Check if T is an airport code for S
# Case 1: T is a subsequence of length 3 from S (converted to uppercase)
if is_subsequence(S, T):
    print("Yes")
# Case 2: First 2 characters of T form a subsequence of S, and the 3rd character is 'X'
elif len(T) == 3 and T[2] == 'X' and is_subsequence(S, T[:2]):
    print("Yes")
else:
    print("No")