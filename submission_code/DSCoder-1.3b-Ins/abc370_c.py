# YOUR CODE HERE

def solve():
    S = input().strip()
    T = input().strip()
    X = []
    while S != T:
        X.append(S)
        S = S[-1] + S[:-1]
    print(len(X) + 1)
    for s in X:
        print(s)

solve()