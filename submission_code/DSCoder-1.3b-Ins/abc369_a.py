# YOUR CODE HERE
A, B = map(int, input().split())

# Initialize count to 0
count = 0

# Loop through possible values of x
for x in range(A, B+1):
    # Check if x is a part of an arithmetic sequence
    if (B - A) == (x - A):
        count += 1

# Print the count
print(count)