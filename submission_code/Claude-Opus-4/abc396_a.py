# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

found = False
for i in range(N - 2):
    if A[i] == A[i+1] == A[i+2]:
        found = True
        break

if found:
    print("Yes")
else:
    print("No")