def find_nth_good_integer(n):
    # Initialize the base number as 0
    base = 0
    # Initialize the count of good integers as 0
    count = 0
    # While the count of good integers is less than n
    while count < n:
        # Increment the base number
        base += 1
        # Convert the base number to a string
        str_base = str(base)
        # Check if all digits in the string are even
        if all(int(digit) % 2 == 0 for digit in str_base):
            # If yes, increment the count of good integers
            count += 1
    # Return the base number
    return base

# Read the input from stdin
n = int(input())
# Find the nth good integer
nth_good_integer = find_nth_good_integer(n)
# Print the nth good integer
print(nth_good_integer)