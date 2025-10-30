# YOUR CODE HERE

def norm(char):
    """
    Normalizes characters 'l' to '1' and 'o' to '0'.
    Other characters are returned unchanged.
    """
    if char == 'l':
        return '1'
    elif char == 'o':
        return '0'
    else:
        return char

# Read input
N = int(input())
S = input()
T = input()

is_similar = True
for i in range(N):
    # Check if the i-th characters (0-indexed) are similar
    # This corresponds to the (i+1)-th characters (1-indexed) in the problem
    if norm(S[i]) != norm(T[i]):
        is_similar = False
        break # Found a difference, no need to check further

# Print the result
if is_similar:
    print("Yes")
else:
    print("No")