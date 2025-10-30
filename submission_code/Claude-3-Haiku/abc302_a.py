# Read the input
A, B = map(int, input().split())

# Calculate the number of attacks needed
attacks = (A + B - 1) // B

# Print the answer
print(attacks)