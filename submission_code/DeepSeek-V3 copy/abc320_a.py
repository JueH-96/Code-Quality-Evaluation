# Read input values for A and B
A, B = map(int, input().split())

# Calculate A^B and B^A
a_power_b = A ** B
b_power_a = B ** A

# Sum the results
result = a_power_b + b_power_a

# Print the result
print(result)