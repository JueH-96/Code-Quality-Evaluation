# YOUR CODE HERE
S = input().strip()

# Extract the number part (last 3 characters)
number_str = S[3:]
number = int(number_str)

# Check if it's a valid contest number
# Valid range: 1-315 and 317-349, excluding 316
if (1 <= number <= 315) or (317 <= number <= 349):
    print("Yes")
else:
    print("No")