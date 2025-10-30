# Read the input
n = int(input())

# The full pi value up to 100 decimal places
pi_full = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"

# Split into integer and decimal parts
parts = pi_full.split('.')
integer_part = parts[0]
decimal_part = parts[1]

# Truncate the decimal part to N digits and construct the result
result = f"{integer_part}.{decimal_part[:n]}"

# Print the result
print(result)