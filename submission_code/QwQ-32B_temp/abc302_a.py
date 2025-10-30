# Read the input values
A, B = map(int, input().split())

# Calculate the minimum number of attacks needed
attacks = (A + B - 1) // B

# Output the result
print(attacks)