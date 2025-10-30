# YOUR CODE HERE
S = input().strip()

# Extract the number part (last 3 characters)
number = int(S[3:])

# Check if it's in the valid range [1, 349] but not 316
if 1 <= number <= 349 and number != 316:
    print("Yes")
else:
    print("No")