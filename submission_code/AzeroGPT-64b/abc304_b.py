import math

def truncate_number(num):
    if num < 10**3:
        return num
    elif num < 10**4:
        return num // 10 * 10
    elif num < 10**5:
        return num // 100 * 100
    elif num < 10**6:
        return num // 1000 * 1000
    elif num < 10**7:
        return num // 10**4 * 10**4
    elif num < 10**8:
        return num // 10**5 * 10**5
    # If number is between 10^8 and 10^9-1, inclusive
    else:
        return num // 10**6 * 10**6

# Read the input from Standard Input
num = int(input())

# Calculate the truncation
truncated_num = truncate_number(num)

# Print the truncated number
print(truncated_num)