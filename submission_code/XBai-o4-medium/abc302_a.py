# Read input values
A, B = map(int, input().split())

# Calculate the minimum number of attacks using ceiling division
attacks = (A + B - 1) // B

# Output the result
print(attacks)