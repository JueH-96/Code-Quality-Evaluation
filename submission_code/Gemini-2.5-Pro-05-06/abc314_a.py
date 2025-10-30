# YOUR CODE HERE
def solve():
    # The given value of pi to the 100-th decimal place
    pi_str = "3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679"

    # Read N from standard input
    N = int(input())

    # The output string needs to include "3." (2 characters) and then N decimal places.
    # So, the total length of the string to be printed is 2 + N.
    # String slicing s[0:L] gives the first L characters of s.
    length_of_output_string = 2 + N
    
    # Truncate pi_str to the required length
    result = pi_str[0:length_of_output_string]

    # Print the result
    print(result)

if __name__ == "__main__":
    solve()