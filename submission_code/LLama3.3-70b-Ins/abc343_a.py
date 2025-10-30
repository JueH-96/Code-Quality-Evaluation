# Read the input from stdin
A, B = map(int, input().split())

# Calculate the sum of A and B
sum_AB = A + B

# Print any integer between 0 and 9, inclusive, that is not equal to A + B
for i in range(10):
    if i != sum_AB:
        print(i)
        break