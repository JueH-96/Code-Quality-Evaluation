# YOUR CODE HERE
A, B = map(int, input().split())
result = A + B

for i in range(10):
    if i != result:
        print(i)
        break