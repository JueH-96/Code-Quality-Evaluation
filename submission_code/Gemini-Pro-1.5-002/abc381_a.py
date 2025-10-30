# YOUR CODE HERE
n = int(input())
s = input()

if n % 2 == 0:
    print("No")
else:
    mid = (n + 1) // 2 - 1
    flag = True
    for i in range(mid):
        if s[i] != '1':
            flag = False
            break
    if s[mid] != '/':
        flag = False
    for i in range(mid + 1, n):
        if s[i] != '2':
            flag = False
            break
    if flag:
        print("Yes")
    else:
        print("No")