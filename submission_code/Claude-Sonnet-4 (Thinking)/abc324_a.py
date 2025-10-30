# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

if all(x == A[0] for x in A):
    print("Yes")
else:
    print("No")