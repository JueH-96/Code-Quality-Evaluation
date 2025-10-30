# YOUR CODE HERE

N = int(input().strip())
A = list(map(int, input().strip().split()))

# Initialize the scores
scores = [0]*N

# Play the games
for i in range(N):
    for j in range(i+1, N):
        if A[i] > A[j]:
            scores[i] += 1
        elif A[i] < A[j]:
            scores[j] -= 1

# Print the final score
print(sum(scores))