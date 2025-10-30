# Read the input integer N
N = int(input())

# Calculate N/5. In Python 3, this is float division.
quotient = N / 5

# Round the quotient to the nearest integer.
# Since N is an integer, N/5 will never be exactly X.5,
# so standard rounding behavior applies.
# round() with one argument returns an int.
rounded_quotient = round(quotient)

# Multiply by 5 to get the position of the nearest water station.
nearest_station = rounded_quotient * 5

# Print the result.
print(nearest_station)