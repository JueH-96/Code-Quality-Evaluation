# Read the input from stdin
n = int(input())

# Convert the number to a string and count the occurrences of each digit
digit_counts = [0] * 10
for digit in str(n):
    digit_counts[int(digit)] += 1

# Check if the conditions are satisfied
if digit_counts[1] == 1 and digit_counts[2] == 2 and digit_counts[3] == 3:
    print("Yes")
else:
    print("No")