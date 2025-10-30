# YOUR CODE HERE
N, D = map(int, input().split())
S = list(input().strip())

# For D days, remove the rightmost cookie
for _ in range(D):
    # Find the rightmost '@'
    for i in range(N-1, -1, -1):
        if S[i] == '@':
            S[i] = '.'
            break

# Convert back to string and print
print(''.join(S))