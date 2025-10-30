# YOUR CODE HERE

N = int(input().strip())
S = input().strip()

# Initialize the count of games Takahashi can win
count = 0

# Iterate over the string S
for i in range(N):
    # If Aoki's move is different from Takahashi's move, increment the count
    if i > 0 and S[i] != S[i-1]:
        count += 1

print(count)