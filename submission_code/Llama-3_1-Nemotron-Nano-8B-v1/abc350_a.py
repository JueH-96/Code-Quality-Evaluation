# Read the input string
S = input().strip()

# Extract the numeric part and convert to integer
num = int(S[3:])

# Check if the number is in the valid range and not 316
if 1 <= num <= 349 and num != 316:
    print("Yes")
else:
    print("No")