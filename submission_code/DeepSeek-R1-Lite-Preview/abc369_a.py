# Read input values
A, B = map(int, input().split())

# Initialize a set to store possible x values
possible_x = set()

# Calculate x1 and x2
x1 = 2 * B - A
x2 = 2 * A - B

# Add x1 and x2 to the set
possible_x.add(x1)
possible_x.add(x2)

# Check if A + B is even to calculate x3
if (A + B) % 2 == 0:
    x3 = (A + B) // 2
    possible_x.add(x3)

# Output the number of distinct x values
print(len(possible_x))