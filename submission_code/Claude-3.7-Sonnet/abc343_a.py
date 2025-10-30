# Read A and B from standard input
A, B = map(int, input().split())

# Calculate their sum
sum_AB = A + B

# Choose a number from 0-9 that isn't equal to the sum
# A simple approach: if sum is 0, print 1, otherwise print 0
if sum_AB == 0:
    print(1)
else:
    print(0)