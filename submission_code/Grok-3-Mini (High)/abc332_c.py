import sys
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1
S = data[index]

# Initialize maximum values
max_A = 0
max_B = 0

# Iterate through the string to find blocks of '1' and '2'
i = 0
while i < N:
    if S[i] == '0':
        i += 1  # Skip days with no plans
    else:
        # Start of a block with plans
        count_A = 0  # Number of days with T-shirt worn
        count_B = 0  # Number of days requiring logo T-shirt
        # Consume the entire block
        while i < N and S[i] != '0':
            if S[i] == '2':
                count_B += 1
            count_A += 1  # Increment for each day with wear
            i += 1  # Move to the next day
        # Update maximum values
        if count_A > 0:  # This should always be true, but safe check
            max_A = max(max_A, count_A)
            max_B = max(max_B, count_B)

# Calculate the minimum number of logo T-shirts to buy
K = max(max_B, max(0, max_A - M))

# Output the result
print(K)