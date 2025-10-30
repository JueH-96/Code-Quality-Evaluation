# YOUR CODE HERE

N = int(input())
S = input()
Q = int(input())

for _ in range(Q):
    t, x, c = map(int, input().split())
    if t == 1:
        S = S[:x] + chr(ord(S[x]) + c) + S[x+1:]
    elif t == 2:
        S = S.lower()
    else:
        S = S.upper()

print(S)