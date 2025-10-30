def max_product_after_increment(n, digits):
    # Find the smallest digit that is not 9 and increment it
    min_digit = min([d for d in digits if d < 9], default=None)
    if min_digit is not None:
        index = digits.index(min_digit)
        digits[index] += 1
    # Calculate the product of all digits
    product = 1
    for digit in digits:
        product *= digit
    return product

# Read the number of test cases
t = int(input().strip())

# Process each test case
for _ in range(t):
    # Read the number of digits
    n = int(input().strip())
    # Read the digits
    digits = list(map(int, input().strip().split()))
    # Output the maximum product
    print(max_product_after_increment(n, digits))