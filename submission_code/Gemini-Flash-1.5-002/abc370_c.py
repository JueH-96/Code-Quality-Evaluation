# YOUR CODE HERE
S = input()
T = input()
n = len(S)
X = []
while S != T:
    best_s = ""
    min_diff = float('inf')
    for i in range(n):
        if S[i] != T[i]:
            temp_s = list(S)
            temp_s[i] = T[i]
            temp_s = "".join(temp_s)
            diff = sum(1 for j in range(n) if temp_s[j] != T[j])
            if diff < min_diff:
                min_diff = diff
                best_s = temp_s
            elif diff == min_diff:
                best_s = min(best_s, temp_s)
    X.append(best_s)
    S = best_s
print(len(X))
for s in X:
    print(s)