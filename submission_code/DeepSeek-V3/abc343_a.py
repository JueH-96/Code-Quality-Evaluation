# YOUR CODE HERE
A, B = map(int, input().split())
total = A + B
for i in range(10):
    if i != total:
        print(i)
        break