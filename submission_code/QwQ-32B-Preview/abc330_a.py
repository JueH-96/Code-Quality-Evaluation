# Read the first line and split into N and L
first_line = input()
parts = first_line.split()
N = int(parts[0])
L = int(parts[1])

# Read the second line and split into a list of integers
scores = list(map(int, input().split()))

# Initialize count
count = 0

# Iterate through the scores and count how many are >= L
for score in scores:
    if score >= L:
        count += 1

# Print the result
print(count)