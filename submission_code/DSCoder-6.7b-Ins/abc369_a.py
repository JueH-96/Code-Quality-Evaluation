# YOUR CODE HERE

# Read inputs
A, B = map(int, input().split())

# Initialize count
count = 0

# Check all possible values of x
for x in range(1, 101):
    # Check if x, A, B can form an arithmetic sequence
    if (A <= x <= B or B <= x <= A) and (A <= B <= x or x <= B <= A) and (B <= x <= A or A <= x <= B):
        count += 1

# Print the count
print(count)