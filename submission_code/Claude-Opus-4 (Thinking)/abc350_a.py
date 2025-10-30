# YOUR CODE HERE
S = input().strip()
# Extract the numeric part (last 3 characters)
num = int(S[3:])

# Check if the number is valid
# Valid ranges: 1-315 and 317-349 (316 is excluded)
if (1 <= num <= 315) or (317 <= num <= 349):
    print("Yes")
else:
    print("No")