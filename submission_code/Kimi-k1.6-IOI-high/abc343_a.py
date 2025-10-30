# Read the input values
A, B = map(int, input().split())

# Calculate the sum of A and B
sum_ab = A + B

# Iterate through numbers 0 to 9 and print the first one not equal to sum_ab
for i in range(10):
    if i != sum_ab:
        print(i)
        break