# Read the input values
A, B = map(int, input().split())

# Initialize a set to store the valid x values
solutions = set()

# Calculate possible x values when x is the first or third term
x2 = 2 * A - B
x3 = 2 * B - A
solutions.add(x2)
solutions.add(x3)

# Calculate x when it is the middle term if A + B is even
if (A + B) % 2 == 0:
    x1 = (A + B) // 2
    solutions.add(x1)

# Output the number of distinct valid x values
print(len(solutions))