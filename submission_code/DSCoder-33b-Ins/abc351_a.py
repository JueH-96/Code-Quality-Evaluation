# YOUR CODE HERE

# Read the inputs
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Calculate the total scores
Takahashi_total = sum(A)
Aoki_total = sum(B)

# Calculate the minimum runs needed for Aoki to win
min_runs = Takahashi_total - Aoki_total + 1

# Print the result
print(min_runs)