# YOUR CODE HERE
N = int(input().strip())
N = str(N)
flag = True
for i in range(len(N)-1):
    if N[i] <= N[i+1]:
        flag = False
        break
if flag:
    print("Yes")
else:
    print("No")