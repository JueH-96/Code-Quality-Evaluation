# YOUR CODE HERE
def are_similar(c1, c2):
    # Same character
    if c1 == c2:
        return True
    # One is '1' and other is 'l'
    if (c1 == '1' and c2 == 'l') or (c1 == 'l' and c2 == '1'):
        return True
    # One is '0' and other is 'o'
    if (c1 == '0' and c2 == 'o') or (c1 == 'o' and c2 == '0'):
        return True
    return False

N = int(input())
S = input().strip()
T = input().strip()

similar = True
for i in range(N):
    if not are_similar(S[i], T[i]):
        similar = False
        break

if similar:
    print("Yes")
else:
    print("No")