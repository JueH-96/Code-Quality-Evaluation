# Read the input values
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Calculate the sum of runs for each team
sumA = sum(A)
sumB = sum(B)

# Compute the required runs for Team Aoki to win
required = sumA - sumB + 1

# Output the result
print(required)