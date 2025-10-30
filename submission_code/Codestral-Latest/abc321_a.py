# YOUR CODE HERE
N = int(input().strip())

# Convert the number to a string to easily access each digit
digits = str(N)

# Check if the digits are strictly decreasing
is_321_like = True
for i in range(len(digits) - 1):
    if digits[i] <= digits[i + 1]:
        is_321_like = False
        break

# Print the result
if is_321_like:
    print("Yes")
else:
    print("No")