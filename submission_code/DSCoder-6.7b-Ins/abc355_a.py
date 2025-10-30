# YOUR CODE HERE
A, B = map(int, input().split())

if A == B:
    print(-1)
else:
    for i in range(1, 4):
        if i != A and i != B:
            print(i)
            break