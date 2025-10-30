# YOUR CODE HERE
S = input().strip()
all_zero = True
for i in range(1, 16, 2):
    if S[i] != '0':
        all_zero = False
        break
print("Yes" if all_zero else "No")