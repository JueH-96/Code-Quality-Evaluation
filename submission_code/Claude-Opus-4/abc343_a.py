# YOUR CODE HERE
A, B = map(int, input().split())
sum_ab = A + B

for i in range(10):
    if i != sum_ab:
        print(i)
        break