# YOUR CODE HERE
N, K = map(int, input().split())
S = list(input().strip())

count = 0
i = 0

while i <= N - K:
    # Check if we have K consecutive healthy teeth starting at position i
    if all(S[j] == 'O' for j in range(i, i + K)):
        # Eat strawberry - make these K teeth unhealthy
        for j in range(i, i + K):
            S[j] = 'X'
        count += 1
        # Move to the next position after the eaten teeth
        i += K
    else:
        # Move to next position
        i += 1

print(count)