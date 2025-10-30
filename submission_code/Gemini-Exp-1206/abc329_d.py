def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    
    counts = [0] * n
    winner = 0
    max_count = 0
    
    for vote in a:
        counts[vote - 1] += 1
        
        if counts[vote - 1] > max_count:
            max_count = counts[vote - 1]
            winner = vote
        elif counts[vote - 1] == max_count and vote < winner:
            winner = vote
            
        print(winner)

solve()