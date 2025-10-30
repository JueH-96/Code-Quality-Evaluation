# YOUR CODE HERE
k = int(input())
count = 0
ans = 0
for i in range(1, 1000000):
    s = str(i)
    flag = True
    for j in range(len(s) - 1):
        if s[j] <= s[j + 1]:
            flag = False
            break
    if flag:
        count += 1
    if count == k:
        ans = i
        break
print(ans)