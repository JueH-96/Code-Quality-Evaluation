# Read the input from stdin
N = int(input())

# The value of pi to the 100th decimal place
pi_to_100th = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"

# Truncate pi to the N-th decimal place and print
print(pi_to_100th[:N+2])