# Read the input for Team Takahashi's scores (9 innings)
A = list(map(int, input().split()))
sum_A = sum(A)

# Read the input for Team Aoki's scores (8 innings)
B = list(map(int, input().split()))
sum_B = sum(B)

# Calculate the minimum required runs for Aoki to win
required = sum_A - sum_B + 1

# Output the result
print(required)