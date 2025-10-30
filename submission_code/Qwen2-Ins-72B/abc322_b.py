N, M = map(int, input().split())
S = input()
T = input()

if S == T[:N] and S == T[-N:]:
    print(0)
elif S == T[:N]:
    print(1)
elif S == T[-N:]:
    print(2)
else:
    print(3)