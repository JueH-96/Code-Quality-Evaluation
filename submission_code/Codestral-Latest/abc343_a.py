# YOUR CODE HERE
A, B = map(int, input().split())
sum_AB = A + B

for i in range(10):
    if i != sum_AB:
        print(i)
        break