# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

S = data[0]
T = data[1]

if S == T:
    print(0)
else:
    M = 0
    X = []
    for i in range(len(S)):
        if S[i] != T[i]:
            X.append(S[:i] + T[i] + S[i+1:])
            M += 1
            S = X[-1]
    X.append(T)
    M += 1
    print(M)
    for s in X:
        print(s)