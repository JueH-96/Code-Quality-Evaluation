def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    A = list(map(int, data[2:2+M]))
    strings = data[2+M:]
    
    # Precompute each player's current score
    curr_scores = []
    for i in range(N):
        s = strings[i]
        score = 0
        for j in range(M):
            if s[j] == 'o':
                score += A[j]
        score += (i + 1)  # bonus
        curr_scores.append(score)
    
    # For each player, compute how many problems they need to solve
    for i in range(N):
        # Find the maximum current score of the other players
        max_other = max(curr_scores[:i] + curr_scores[i+1:])
        
        # If player i's current score is already strictly greater than 
        # all other players' current scores, print 0
        if curr_scores[i] > max_other:
            print(0)
            continue
        
        # Otherwise, list the scores of the unsolved problems for player i
        s = strings[i]
        unsolved_scores = [
            A[j] for j in range(M) if s[j] == 'x'
        ]
        
        # Sort the scores of unsolved problems in descending order 
        # to solve the largest ones first
        unsolved_scores.sort(reverse=True)
        
        # Start accumulating from player i's current score
        needed = 0
        current = curr_scores[i]
        # Keep adding scores of unsolved problems until the player's score
        # exceeds the maximum of other players' current scores
        while current <= max_other:
            current += unsolved_scores[needed]
            needed += 1
        
        # Print the number of problems needed
        print(needed)

# Call solve() after definition
if __name__ == "__main__":
    solve()