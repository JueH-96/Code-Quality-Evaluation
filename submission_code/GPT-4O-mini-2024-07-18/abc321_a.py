def is_321_like_number(n):
    digits = list(str(n))
    for i in range(len(digits) - 1):
        if digits[i] <= digits[i + 1]:
            return "No"
    return "Yes"

# Read input
N = int(input().strip())
# Print the result
print(is_321_like_number(N))