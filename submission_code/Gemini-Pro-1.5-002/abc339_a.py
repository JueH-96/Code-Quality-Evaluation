# YOUR CODE HERE
s = input()
n = len(s)
ans = ""
for i in range(n - 1, -1, -1):
    if s[i] == '.':
        break
    ans = s[i] + ans
print(ans)