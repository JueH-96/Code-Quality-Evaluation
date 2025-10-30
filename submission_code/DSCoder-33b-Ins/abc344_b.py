# YOUR CODE HERE

A = []
while True:
    a = int(input())
    if a == 0:
        A.append(a)
        break
    A.append(a)

for i in reversed(A):
    print(i)