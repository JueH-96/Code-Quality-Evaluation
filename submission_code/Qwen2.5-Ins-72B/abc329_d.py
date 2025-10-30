# YOUR CODE HERE
N, M = map(int, input().split())
A = list(map(int, input().split()))

votes = [0] * (N + 1)
current_winner = 0
max_votes = 0

for i in range(M):
    votes[A[i]] += 1
    if votes[A[i]] > max_votes or (votes[A[i]] == max_votes and A[i] < current_winner):
        max_votes = votes[A[i]]
        current_winner = A[i]
    print(current_winner)