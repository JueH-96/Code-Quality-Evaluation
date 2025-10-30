# YOUR CODE HERE
N, M, K = map(int, input().split())
A = list(map(int, input().split()))

total_votes = sum(A)
remaining_votes = K - total_votes

C = [0] * N

for i in range(N):
    votes_needed = M - (total_votes - A[i] + 1)
    if votes_needed > 0:
        if votes_needed > remaining_votes:
            C[i] = -1
        else:
            C[i] = votes_needed

print(*C)