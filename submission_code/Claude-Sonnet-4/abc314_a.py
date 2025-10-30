# YOUR CODE HERE
N = int(input())

# Pi to 100 decimal places
pi_str = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"

# Extract the part we need: "3." + first N digits after decimal point
result = "3." + pi_str[2:2+N]

print(result)