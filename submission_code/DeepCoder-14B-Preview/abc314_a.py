n = int(input())
pi_str = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"

# Split the pi string into integer and decimal parts
if '.' in pi_str:
    integer_part, decimal_part = pi_str.split('.')
else:
    integer_part = pi_str
    decimal_part = ''

# Take the first N decimal digits
required_decimals = decimal_part[:n]

# Form the result
result = f"{integer_part}.{required_decimals}"
print(result)