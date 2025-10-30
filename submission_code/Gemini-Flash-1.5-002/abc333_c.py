# YOUR CODE HERE
def repunit(n):
    return (10**n - 1) // 9

n = int(input())
ans = []
for i in range(1, 1001):
    for j in range(i, 1001):
        for k in range(j, 1001):
            s = repunit(i) + repunit(j) + repunit(k)
            ans.append(s)
ans = sorted(list(set(ans)))
print(ans[n-1])