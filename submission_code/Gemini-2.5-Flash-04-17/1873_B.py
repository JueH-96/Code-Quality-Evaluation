import sys

# Read the number of test cases
t = int(sys.stdin.readline())

# Process each test case
for _ in range(t):
    # Read the number of digits
    n = int(sys.stdin.readline())
    # Read the digits into a list
    a = list(map(int, sys.stdin.readline().split()))

    # Initialize maximum product found so far
    # Products are always non-negative since digits are >= 0.
    # Starting with 0 is safe.
    max_product = 0

    # Iterate through each digit's index to consider adding 1 to it
    # There are 'n' possible digits to add 1 to.
    for i in range(n):
        # Create a temporary list by copying the original list
        # This is important so we don't modify the original list 'a'
        temp_a = list(a)

        # Increment the digit at index i by 1.
        # This is the one required modification as per the problem.
        temp_a[i] += 1

        # Calculate the product of the modified list
        current_product = 1
        # Iterate through the elements of the modified list
        for digit in temp_a:
            current_product *= digit

        # Update the maximum product found so far
        max_product = max(max_product, current_product)

    # Output the maximum product for the current test case, followed by a newline
    sys.stdout.write(str(max_product) + '
')