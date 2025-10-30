def are_similar_chars(x, y):
    if x == y:
        return True
    if (x == '1' and y == 'l') or (x == 'l' and y == '1'):
        return True
    if (x == '0' and y == 'o') or (x == 'o' and y == '0'):
        return True
    return False

def are_similar_strings(N, S, T):
    for i in range(N):
        if not are_similar_chars(S[i], T[i]):
            return "No"
    return "Yes"

import sys
input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
S = data[1]
T = data[2]

result = are_similar_strings(N, S, T)
print(result)