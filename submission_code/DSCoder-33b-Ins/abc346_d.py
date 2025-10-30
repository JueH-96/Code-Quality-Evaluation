# YOUR CODE HERE

N = int(input())
S = input()
C = list(map(int, input().split()))

# Initialize the cost to 0
cost = 0

# Iterate over the string S
for i in range(1, N):
    # If the current character is the same as the previous one
    if S[i] == S[i-1]:
        # If the cost of changing the current character is less than the cost of changing the previous character
        if C[i] < C[i-1]:
            # Change the current character to the opposite
            S = S[:i] + str(1 - int(S[i])) + S[i+1:]
            # Add the cost of changing the current character to the total cost
            cost += C[i]
        else:
            # Change the previous character to the opposite
            S = S[:i-1] + str(1 - int(S[i-1])) + S[i:]
            # Add the cost of changing the previous character to the total cost
            cost += C[i-1]

# Print the total cost
print(cost)