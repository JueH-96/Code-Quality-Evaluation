# Define the string for pi up to 100 decimal places
pi_str = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"

# Read the integer N from standard input
N = int(input())

# Print the truncated pi string up to N decimal places
print(pi_str[:2 + N])