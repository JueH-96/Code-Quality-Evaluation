def are_similar_characters(x, y):
    if x == y:
        return True
    if (x == '1' and y == 'l') or (x == 'l' and y == '1'):
        return True
    if (x == '0' and y == 'o') or (x == 'o' and y == '0'):
        return True
    return False

def are_similar_strings(S, T):
    N = len(S)
    for i in range(N):
        if not are_similar_characters(S[i], T[i]):
            return False
    return True

N = int(input())
S = input()
T = input()

if are_similar_strings(S, T):
    print("Yes")
else:
    print("No")