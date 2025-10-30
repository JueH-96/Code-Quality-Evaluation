# Read the input from stdin
A, B = map(int, input().split())

# Calculate the sum of A and B
sum_ab = A + B

# Find any number between 0 and 9 that is not equal to A + B
for i in range(10):
    if i != sum_ab:
        print(i)
        break