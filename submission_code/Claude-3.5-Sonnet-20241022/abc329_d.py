N, M = map(int, input().split())
A = list(map(int, input().split()))

votes = [0] * (N+1)  # votes[i] = number of votes for candidate i

for i in range(M):
    votes[A[i]] += 1
    
    # Find candidate with most votes so far
    max_votes = 0
    winner = 1
    for j in range(1, N+1):
        if votes[j] > max_votes:
            max_votes = votes[j]
            winner = j
        elif votes[j] == max_votes and j < winner:
            winner = j
            
    print(winner)