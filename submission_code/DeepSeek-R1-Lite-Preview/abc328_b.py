# Read the number of months
N = int(input())

# Read the list of days in each month
D = list(map(int, input().split()))

count = 0

# Iterate through each month
for i in range(1, N + 1):
    # Iterate through each day in the month
    for j in range(1, D[i - 1] + 1):
        # Convert month and day to strings
        str_i = str(i)
        str_j = str(j)
        # Combine the strings
        combined = str_i + str_j
        # Check if all characters in the combined string are the same
        if len(set(combined)) == 1:
            count += 1

# Print the total count of repdigit dates
print(count)