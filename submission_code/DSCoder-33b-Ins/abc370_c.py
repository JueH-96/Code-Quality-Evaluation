# YOUR CODE HERE

def solve(S, T):
    if S == T:
        return [0]
    else:
        X = []
        for i in range(len(S)):
            if S[i] != T[i]:
                S = S[:i] + T[i] + S[i+1:]
                X.append(S)
        return [len(X)] + X

S = input()
T = input()
result = solve(S, T)
for r in result:
    print(r)