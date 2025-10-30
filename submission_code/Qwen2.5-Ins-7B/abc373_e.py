# N M K
# A_1 A_2 ... A_N
N, M, K = map(int, input().split())
A = list(map(int, input().split()))

# Calculate the total number of votes
total_votes = sum(A)

# Calculate the number of votes needed to be ahead of M-1 candidates
votes_needed = (N - M + 1) * (total_votes // (N - M + 1) + 1)

# Calculate the minimum additional votes needed for each candidate
C = []
for i in range(N):
    if A[i] + K - total_votes >= votes_needed:
        C.append(0)
    else:
        additional_votes_needed = votes_needed - A[i] - (K - total_votes - 1)
        if additional_votes_needed > K - total_votes:
            C.append(-1)
        else:
            C.append(additional_votes_needed)

# Print the result
print(*C)