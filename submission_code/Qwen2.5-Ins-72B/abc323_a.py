# YOUR CODE HERE
S = input().strip()
for i in range(1, 16, 2):
    if S[i] != '0':
        print("No")
        break
else:
    print("Yes")