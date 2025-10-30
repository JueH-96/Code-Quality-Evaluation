# YOUR CODE HERE
N = int(input().strip())

# Convert the number to a string to easily count each digit
digits = str(N)

# Count the occurrences of each digit
count_1 = digits.count('1')
count_2 = digits.count('2')
count_3 = digits.count('3')

# Check if the counts match the required conditions
if count_1 == 1 and count_2 == 2 and count_3 == 3:
    print("Yes")
else:
    print("No")