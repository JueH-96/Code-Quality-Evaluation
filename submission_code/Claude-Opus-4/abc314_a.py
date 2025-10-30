# YOUR CODE HERE
N = int(input())

# Pi to 100 decimal places as given in the problem
pi_str = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"

# Find the position to truncate
# We need the decimal point plus N decimal places
# Position 0 is '3', position 1 is '.', so we need positions 0 to 1+N
truncated_pi = pi_str[:2+N]

print(truncated_pi)