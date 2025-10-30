# YOUR CODE HERE
N = int(input())
S = input()

# Initialize the sum to 0
sum = 0

# Iterate over the string
for i in range(N):
    # If the current character is 1, add 1 to the sum
    if S[i] == '1':
        sum += 1

# Print the sum
print(sum)