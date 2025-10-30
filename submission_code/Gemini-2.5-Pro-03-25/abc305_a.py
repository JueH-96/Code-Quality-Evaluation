# YOUR CODE HERE
import sys

# Read the input integer N from standard input
# N represents Takahashi's position in kilometers.
N = int(sys.stdin.readline())

# Water stations are located at every 5 km, i.e., at positions 0, 5, 10, ..., 100.
# These are multiples of 5.
# We need to find the multiple of 5 that is closest to N.

# One way to find the nearest multiple of 5 is to:
# 1. Divide N by 5.
# 2. Round the result to the nearest integer.
# 3. Multiply the rounded result by 5.

# Calculate N / 5
quotient = N / 5

# Round the quotient to the nearest integer.
# In Python 3, round() implements "round half to even".
# For example, round(2.5) is 2, and round(3.5) is 4.
# However, since N is an integer, N/5 can never result in a value
# exactly halfway between two integers (like k.5).
# The possible fractional parts are .0, .2, .4, .6, .8.
# For these fractional parts, round() behaves as standard rounding:
# round(k.0) = k
# round(k.2) = k
# round(k.4) = k
# round(k.6) = k + 1
# round(k.8) = k + 1
# This correctly identifies the nearest integer multiple of the quotient.
rounded_quotient = round(quotient)

# Multiply the rounded quotient by 5 to get the position of the nearest water station.
# We cast to int to ensure the output is strictly an integer type, although the result
# of round(N/5) * 5 should mathematically be an integer when N is an integer.
nearest_station_position = int(rounded_quotient * 5)

# Print the calculated position of the nearest water station to standard output.
print(nearest_station_position)

# YOUR CODE HERE