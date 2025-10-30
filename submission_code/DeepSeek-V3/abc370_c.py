# YOUR CODE HERE
S = input().strip()
T = input().strip()
n = len(S)
X = []
current_S = list(S)
for i in range(n):
    if current_S[i] != T[i]:
        current_S[i] = T[i]
        X.append(''.join(current_S))
print(len(X))
for s in X:
    print(s)