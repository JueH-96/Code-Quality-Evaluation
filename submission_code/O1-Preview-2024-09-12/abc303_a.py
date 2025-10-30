# YOUR CODE HERE
N = int(input())
S = input()
T = input()

def similar_char(x, y):
    if x == y:
        return True
    elif (x == '1' and y == 'l') or (x == 'l' and y == '1'):
        return True
    elif (x == '0' and y == 'o') or (x == 'o' and y == '0'):
        return True
    else:
        return False

similar = True
for i in range(N):
    if not similar_char(S[i], T[i]):
        similar = False
        break

if similar:
    print("Yes")
else:
    print("No")