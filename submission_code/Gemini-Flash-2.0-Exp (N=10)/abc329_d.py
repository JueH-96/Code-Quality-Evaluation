def solve():
    n, m = map(int, input().split())
    votes = list(map(int, input().split()))
    
    counts = [0] * (n + 1)
    
    for i in range(m):
        counts[votes[i]] += 1
        
        max_votes = 0
        winner = -1
        
        for j in range(1, n + 1):
            if counts[j] > max_votes:
                max_votes = counts[j]
                winner = j
            elif counts[j] == max_votes and j < winner:
                winner = j
        print(winner)

solve()