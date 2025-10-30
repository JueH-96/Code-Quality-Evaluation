# Read the two integers A and B from standard input.
# The input is space-separated, so we use split() and map() to parse them.
A, B = map(int, input().split())

# Calculate the sum of A and B.
# The problem constraints guarantee 0 <= A + B <= 9.
forbidden_number = A + B

# We need to find any integer from 0 to 9 that is not equal to the sum.
# A simple loop can accomplish this. We iterate through the numbers 0-9.
for i in range(10):
    # Check if the current number is the one we are not allowed to print.
    if i != forbidden_number:
        # If it's not the forbidden number, print it.
        print(i)
        # Since we only need to print one such integer, we can exit the loop.
        break