import sys

def is_321_like_number(n):
    # Convert the number to a string to easily access each digit
    digits = str(n)
    # Iterate through the digits and check if each digit is greater than the next one
    for i in range(len(digits) - 1):
        if digits[i] <= digits[i + 1]:
            return "No"
    return "Yes"

# Read input from stdin
input = sys.stdin.read().strip()
N = int(input)

# Check if N is a 321-like Number and print the result
result = is_321_like_number(N)
print(result)