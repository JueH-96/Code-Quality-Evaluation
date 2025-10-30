# YOUR CODE HERE
A, B = map(int, input().split())
sum_AB = A + B

# Find any number from 0-9 that is not equal to A + B
for i in range(10):
    if i != sum_AB:
        print(i)
        break