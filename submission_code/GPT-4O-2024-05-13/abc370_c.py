# YOUR CODE HERE
def find_minimal_transformations(S, T):
    n = len(S)
    X = []
    S = list(S)
    T = list(T)
    
    for i in range(n):
        if S[i] != T[i]:
            S[i] = T[i]
            X.append("".join(S))
    
    print(len(X))
    for s in X:
        print(s)

import sys
input = sys.stdin.read
data = input().split()
S = data[0]
T = data[1]

find_minimal_transformations(S, T)