import sys

# The number pi to the 100-th decimal place as a string
PI_100_DECIMALS = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"

def solve():
    # Read the integer N from standard input
    N = int(sys.stdin.readline())

    # The string representation of pi starts with "3."
    # To get N decimal places, we need the "3." (2 characters)
    # plus the N characters after the decimal point.
    # So, the total length of the substring needed is 2 + N.
    output_length = 2 + N

    # Slice the PI_100_DECIMALS string to get the desired truncation.
    # Python string slicing s[start:end] includes characters from start up to (but not including) end.
    # So, s[0:output_length] will give exactly the first 'output_length' characters.
    truncated_pi = PI_100_DECIMALS[0:output_length]

    # Print the result
    print(truncated_pi)

if __name__ == '__main__':
    solve()