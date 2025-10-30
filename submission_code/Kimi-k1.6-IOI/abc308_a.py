# Read the input as a list of integers
s = list(map(int, input().split()))

# Initialize a flag to track validity
valid = True

# Check each number for the second and third conditions
for num in s:
    if not (100 <= num <= 675) or num % 25 != 0:
        valid = False
        break

# If all numbers are valid, check the first condition (non-decreasing)
if valid:
    for i in range(7):
        if s[i] > s[i+1]:
            valid = False
            break

# Print the result
print("Yes" if valid else "No")