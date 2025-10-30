N, M = map(int, input().split())
A = list(map(int, input().split()))

count = [0] * (N + 1)  # 1-indexed
max_votes = 0
winner = 1

for i in range(M):
    candidate = A[i]
    count[candidate] += 1
    
    # Check if this candidate should be the new winner
    if count[candidate] > max_votes:
        max_votes = count[candidate]
        winner = candidate
    elif count[candidate] == max_votes and candidate < winner:
        winner = candidate
    
    print(winner)