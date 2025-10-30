import sys

# Read all input from stdin
data = sys.stdin.read().split()

# Read N
index = 0
N = int(data[index])
index += 1

# Read the array A
A = list(map(int, data[index:index + N]))

# Initialize count of operations
count = 0

# Loop until there is one or fewer positive elements
while True:
    # Count the number of positive elements
    pos_count = sum(1 for x in A if x > 0)
    if pos_count <= 1:
        break
    # Sort the array in descending order
    A.sort(reverse=True)
    # Decrease the first two elements by 1
    A[0] -= 1
    A[1] -= 1
    # Increment the operation count
    count += 1

# Output the number of operations
print(count)