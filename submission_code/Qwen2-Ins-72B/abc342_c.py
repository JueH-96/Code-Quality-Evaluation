N = int(input())
S = list(input())
Q = int(input())

operations = [list(input().split()) for _ in range(Q)]

char_map = {char: char for char in S}

for c, d in reversed(operations):
    char_map[c] = char_map[d]

for i in range(N):
    S[i] = char_map[S[i]]

print("".join(S))