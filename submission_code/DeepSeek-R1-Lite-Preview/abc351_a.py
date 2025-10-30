# Read the inputs
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Calculate total runs for Takahashi and Aoki
S_T = sum(A)
S_A = sum(B)

# Calculate the minimum runs Aoki needs to score in the ninth inning to win
X = S_T - S_A + 1

# Output the result
print(X)