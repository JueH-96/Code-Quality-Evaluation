import sys

S, T = input().split()

for c in range(1, len(S)+1):
    for w in range(c, len(S)):
        result = ''.join([S[i::w] for i in range(c)])
        if result == T:
            print("Yes")
            sys.exit()

print("No")