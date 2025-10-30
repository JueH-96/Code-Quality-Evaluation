# YOUR CODE HERE
a, b, d = map(int, input().split())
ans = []
for i in range(a, b + 1, d):
    ans.append(str(i))
print(' '.join(ans))