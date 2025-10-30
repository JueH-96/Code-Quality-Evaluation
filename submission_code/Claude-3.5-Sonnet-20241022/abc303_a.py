N = int(input())
S = input()
T = input()

def is_similar(c1, c2):
    if c1 == c2:
        return True
    if (c1 == '1' and c2 == 'l') or (c1 == 'l' and c2 == '1'):
        return True
    if (c1 == '0' and c2 == 'o') or (c1 == 'o' and c2 == '0'):
        return True
    return False

similar = True
for i in range(N):
    if not is_similar(S[i], T[i]):
        similar = False
        break

print("Yes" if similar else "No")