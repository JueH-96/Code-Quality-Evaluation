# Read input values for A and B
A, B = map(int, input().split())

# Calculate the sum of A and B
sum_AB = A + B

# Iterate through numbers from 0 to 9 and find one that is not equal to sum_AB
for i in range(10):
    if i != sum_AB:
        print(i)
        break