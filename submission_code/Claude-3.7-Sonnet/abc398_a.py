# Read input
N = int(input())

# Create a string of all "-" characters initially
result = ["-"] * N

if N % 2 == 0:  # Even length
    # Place two "=" characters in the middle
    result[N//2-1:N//2+1] = ["=", "="]
else:  # Odd length
    # Place one "=" character in the middle
    result[N//2] = "="

# Print the result
print("".join(result))