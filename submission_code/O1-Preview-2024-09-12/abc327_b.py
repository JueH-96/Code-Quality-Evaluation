# YOUR CODE HERE
B = int(input())
for A in range(1, 70):
    if pow(A, A) == B:
        print(A)
        break
else:
    print(-1)