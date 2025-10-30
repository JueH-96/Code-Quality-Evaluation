# YOUR CODE HERE
N = int(input())
S = input()

found = False
for i in range(N - 1):
    if (S[i] == 'a' and S[i + 1] == 'b') or (S[i] == 'b' and S[i + 1] == 'a'):
        found = True
        break

if found:
    print("Yes")
else:
    print("No")