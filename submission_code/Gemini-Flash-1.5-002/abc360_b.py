# YOUR CODE HERE
S, T = input().split()
n = len(S)
m = len(T)
ans = "No"
for w in range(1, n):
    temp = ""
    for i in range(0, n, w):
        sub = S[i:i + w]
        if len(sub) >= 1:
            for c in range(1, min(len(sub) + 1, w + 1)):
                if len(sub) >= c:
                    temp += sub[c - 1]
    if temp == T:
        ans = "Yes"
        break
print(ans)