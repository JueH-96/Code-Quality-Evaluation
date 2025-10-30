import sys

def count_snake_numbers(L, R):
    count = 0
    for num in range(L, R+1):
        if num >= 10:
            top_digit = int(str(num)[0])
            other_digits = [int(d) for d in str(num)[1:]]
            if top_digit > max(other_digits):
                count += 1
    return count

# Read input from stdin
L, R = map(int, input().split())

# Solve the problem
result = count_snake_numbers(L, R)

# Write output to stdout
print(result)