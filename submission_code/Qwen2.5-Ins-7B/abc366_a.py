# YOUR CODE HERE
N = int(input())
T = int(input())
A = int(input())

if T > A + (N - T) // 2:
    print("Yes")
else:
    print("No")