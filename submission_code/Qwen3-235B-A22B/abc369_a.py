# Read input
A, B = map(int, input().split())

# Calculate the two x values that are always integers
x2 = 2 * A - B
x3 = 2 * B - A

# Initialize a set with these two values
result_set = {x2, x3}

# Check if the sum of A and B is even to add the third possible x value
if (A + B) % 2 == 0:
    x1 = (A + B) // 2
    result_set.add(x1)

# The answer is the size of the set
print(len(result_set))