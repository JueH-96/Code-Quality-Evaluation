# YOUR CODE HERE
def are_similar_chars(x, y):
    # Check if characters are the same
    if x == y:
        return True
    # Check if one is 1 and other is l
    if (x == '1' and y == 'l') or (x == 'l' and y == '1'):
        return True
    # Check if one is 0 and other is o
    if (x == '0' and y == 'o') or (x == 'o' and y == '0'):
        return True
    return False

# Read input
N = int(input())
S = input().strip()
T = input().strip()

# Check if all characters at corresponding positions are similar
similar = True
for i in range(N):
    if not are_similar_chars(S[i], T[i]):
        similar = False
        break

# Output result
if similar:
    print("Yes")
else:
    print("No")