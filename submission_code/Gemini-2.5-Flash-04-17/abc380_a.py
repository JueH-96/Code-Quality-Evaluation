import sys

# Read the input number as a string
n_str = sys.stdin.readline().strip()

# Count the occurrences of each required digit
count_1 = n_str.count('1')
count_2 = n_str.count('2')
count_3 = n_str.count('3')

# Check if the counts match the required frequencies
if count_1 == 1 and count_2 == 2 and count_3 == 3:
    print("Yes")
else:
    print("No")