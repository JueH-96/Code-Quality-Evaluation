# YOUR CODE HERE

N, M = map(int, input().split())
S = input()
T = input()

if S == T[0:N]:
    if S == T[-N:]:
        print(0)
    else:
        print(1)
elif S == T[-N:]:
    print(2)
else:
    print(3)