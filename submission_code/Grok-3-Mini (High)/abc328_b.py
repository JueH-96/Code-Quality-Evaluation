import sys

# Read all input from stdin
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
days = list(map(int, data[index:index + N]))

# Initialize count of repdigit dates
count = 0

# Iterate over each possible digit D from 1 to 9
for D in range(1, 10):
    # Check for month i = D
    if D <= N:
        i = D
        max_day = days[i - 1]  # Number of days in month i (1-based index)
        # Check for day j = D
        if max_day >= D:
            count += 1  # j = D is a repdigit day
        # Check for day j = 11 * D
        if max_day >= 11 * D:
            count += 1  # j = 11 * D is a repdigit day
    
    # Check for month i = 11 * D
    if 11 * D <= N:
        i = 11 * D
        max_day = days[i - 1]  # Number of days in month i (1-based index)
        # Check for day j = D
        if max_day >= D:
            count += 1  # j = D is a repdigit day
        # Check for day j = 11 * D
        if max_day >= 11 * D:
            count += 1  # j = 11 * D is a repdigit day

# Output the total count
print(count)