# YOUR CODE HERE
S = input().strip()
T = input().strip()

X = []

for i in range(len(S)):
    if S[i] != T[i]:
        S = S[:i] + T[i] + S[i+1:]
        X.append(S)

print(len(X))
for s in X:
    print(s)