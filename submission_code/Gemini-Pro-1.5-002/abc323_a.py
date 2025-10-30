# YOUR CODE HERE
s = input()
ans = "Yes"
for i in range(1, 8):
    if s[2 * i] == '1':
        ans = "No"
        break
print(ans)