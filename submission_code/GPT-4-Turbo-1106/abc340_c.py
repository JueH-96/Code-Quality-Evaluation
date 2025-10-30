def total_payment(N):
    total_cost = 0
    while N >= 2:
        total_cost += N
        N //= 2
    return total_cost

# Read the input from stdin
N = int(input().strip())

# Calculate the total payment
result = total_payment(N)

# Write the answer to stdout
print(result)