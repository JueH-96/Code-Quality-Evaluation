# Read the input integer N
N = int(input())

# Variable to store the calculated answer
ans = 0

# Apply the conditions based on the value of N
if N < 1000:  # Corresponds to N <= 10^3 - 1
    # Print N as it is
    ans = N
elif N < 10000: # Corresponds to 10^3 <= N <= 10^4 - 1
    # Truncate the ones digit (1 digit from right)
    divisor = 10
    ans = (N // divisor) * divisor
elif N < 100000: # Corresponds to 10^4 <= N <= 10^5 - 1
    # Truncate the tens digit and below (2 digits from right)
    divisor = 100
    ans = (N // divisor) * divisor
elif N < 1000000: # Corresponds to 10^5 <= N <= 10^6 - 1
    # Truncate the hundreds digit and below (3 digits from right)
    divisor = 1000
    ans = (N // divisor) * divisor
elif N < 10000000: # Corresponds to 10^6 <= N <= 10^7 - 1
    # Truncate the thousands digit and below (4 digits from right)
    divisor = 10000
    ans = (N // divisor) * divisor
elif N < 100000000: # Corresponds to 10^7 <= N <= 10^8 - 1
    # Truncate the ten-thousands digit and below (5 digits from right)
    divisor = 100000
    ans = (N // divisor) * divisor
else:  # Corresponds to 10^8 <= N <= 10^9 - 1
    # This case handles N values from 100,000,000 up to 999,999,999.
    # Truncate the hundred-thousands digit and below (6 digits from right)
    divisor = 1000000
    ans = (N // divisor) * divisor

# Print the final answer
print(ans)