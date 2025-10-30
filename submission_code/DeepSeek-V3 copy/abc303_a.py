# YOUR CODE HERE
def are_similar_characters(c1, c2):
    if c1 == c2:
        return True
    if (c1 == '1' and c2 == 'l') or (c1 == 'l' and c2 == '1'):
        return True
    if (c1 == '0' and c2 == 'o') or (c1 == 'o' and c2 == '0'):
        return True
    return False

def are_similar_strings(N, S, T):
    for i in range(N):
        if not are_similar_characters(S[i], T[i]):
            return False
    return True

# Read input
N = int(input())
S = input().strip()
T = input().strip()

# Check if S and T are similar strings
if are_similar_strings(N, S, T):
    print("Yes")
else:
    print("No")