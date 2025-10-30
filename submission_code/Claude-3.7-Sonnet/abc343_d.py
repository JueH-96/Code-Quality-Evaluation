def solve():
    N, T = map(int, input().split())
    
    # Initialize scores for all players (0-indexed but players are 1-indexed)
    scores = [0] * N
    
    for _ in range(T):
        a, b = map(int, input().split())
        # Update score of player a
        scores[a-1] += b
        
        # Count the number of unique scores
        unique_scores = len(set(scores))
        print(unique_scores)

solve()