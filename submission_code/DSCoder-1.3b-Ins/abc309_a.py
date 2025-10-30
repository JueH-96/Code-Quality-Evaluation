# YOUR CODE HERE
A, B = map(int, input().split())

if A + 1 == B or A - 1 == B or A == B:
    print("Yes")
else:
    print("No")