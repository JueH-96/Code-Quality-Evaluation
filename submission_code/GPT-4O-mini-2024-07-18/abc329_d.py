def election_winner(N, M, votes):
    vote_count = [0] * (N + 1)
    current_winner = 0
    max_votes = 0
    
    results = []
    
    for i in range(M):
        candidate = votes[i]
        vote_count[candidate] += 1
        
        if vote_count[candidate] > max_votes:
            max_votes = vote_count[candidate]
            current_winner = candidate
        elif vote_count[candidate] == max_votes:
            current_winner = min(current_winner, candidate)
        
        results.append(current_winner)
    
    return results

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
votes = list(map(int, data[2:M+2]))

results = election_winner(N, M, votes)
print('
'.join(map(str, results)))