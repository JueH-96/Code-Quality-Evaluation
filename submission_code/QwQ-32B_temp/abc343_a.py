# Read the input values
A, B = map(int, input().split())
sum_ab = A + B

# Find and print the first number from 0 to 9 that is not equal to the sum
for i in range(10):
    if i != sum_ab:
        print(i)
        break