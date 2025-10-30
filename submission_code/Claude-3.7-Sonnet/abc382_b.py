# YOUR CODE HERE
# Read input
N, D = map(int, input().split())
S = list(input())

# Simulate Takahashi's actions for D days
for _ in range(D):
    # Find the rightmost box containing a cookie
    for i in range(N-1, -1, -1):
        if S[i] == '@':
            S[i] = '.'  # Remove the cookie
            break

# Print the final state of the boxes
print(''.join(S))